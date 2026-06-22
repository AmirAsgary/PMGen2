"""
MHC-Diff (model_2) training entrypoint.

Diffusion denoiser + pLDDT head, trained on the SAME processed H5 store as
model_1 (targets = PMGen/AF2 teacher Cα + teacher pLDDT). Backbone-only: no
side-chain torsion loss (the store has no side chains). See SPEC.md.

  # local smoke on the 15-example dummy set
  python src/model_2/train.py --dummy --epochs 2 --bs 3
  # real run
  python src/model_2/train.py --h5-dir data/processed/h5_store \
      --scheme two_axis --fold 1 --epochs 50 --bs 8 --amp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M                                       # noqa: E402


def setup_distributed():
    """Read torchrun/SLURM env. Returns (distributed, rank, local_rank, world)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world
    return False, 0, 0, 1

_COLS = ["wall_time", "split", "epoch", "step", "lr", "total",
         "pep_coord", "mhc_coord", "torsion", "plddt", "pep_plddt",
         "contain", "pep_ca_rmsd", "mhc_ca_rmsd"]


def _csv_cols(p: Path):
    """Use an existing file's header (so a resumed run stays aligned), else create
    one with the current schema. Returns the column list to write."""
    if p.exists():
        return p.read_text().splitlines()[0].split(",")
    p.write_text(",".join(_COLS) + "\n")
    return list(_COLS)


def _csv_row(p: Path, cols, **kw):
    with open(p, "a") as f:
        f.write(",".join("" if kw.get(c) is None else f"{kw.get(c)}"
                         for c in cols) + "\n")


@torch.no_grad()
def evaluate(net, loader, device, lambdas, max_batches=None):
    net.eval()
    agg, n = {}, 0
    for i, data in enumerate(loader):
        data = data.to(device)
        _, terms = net.compute_loss(data, lambdas)
        bs = int(data.num_graphs)
        for k, v in terms.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
        if max_batches and i + 1 >= max_batches:
            break
    net.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def _superpose_on(p, t, align, evalm):
    """RMSD over evalm points after Kabsch-aligning on the align points."""
    pa, ta = p[align], t[align]
    mp, mt = pa.mean(0), ta.mean(0)
    u, _, vt = torch.linalg.svd((pa - mp).T @ (ta - mt))
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rot = vt.T @ torch.diag(torch.tensor([1., 1., d], device=p.device)) @ u.T
    return float((((p[evalm] - mp) @ rot.T - (t[evalm] - mt)) ** 2).sum(-1).mean().sqrt())


@torch.no_grad()
def sample_rmsd(core, loader, device, n_steps=25, max_graphs=64):
    """Structural eval: DDIM-sample (MHC from its true coords, peptide from noise)
    and report peptide/MHC Cα-RMSD — the number directly comparable to model_1.
    Peptide aligned on the MHC."""
    core.eval()
    peps, mhcs, seen = [], [], 0
    for data in loader:
        data = data.to(device)
        pred = core.sample(data, sampler="ddim", n_steps=n_steps, mhc_from_truth=True)
        for g in range(data.num_graphs):
            m = data.batch == g
            p, t, pep = pred[m], data.pos[m], data.pep[m].bool()
            if not torch.isfinite(p).all():
                continue
            peps.append(_superpose_on(p, t, ~pep, pep))
            mhcs.append(_superpose_on(p, t, ~pep, ~pep))
        seen += data.num_graphs
        if seen >= max_graphs:
            break
    core.train()
    import numpy as np
    return ({"pep_ca_rmsd": float(np.median(peps)), "mhc_ca_rmsd": float(np.median(mhcs)),
             "n": len(peps)} if peps else {})


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-dir", default=None)
    p.add_argument("--scheme", default="two_axis", choices=["two_axis", "hla_only"])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--mhc-scale", type=float, default=0.1,
                   help="MHC noise variance relative to peptide (induced fit)")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--k", type=int, default=16, help="k-NN graph degree")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--no-cross", action="store_true",
                   help="disable the chirality (cross-product) term -> E(3) only")
    p.add_argument("--lambdas", type=float, nargs=4, default=(1.0, 0.25, 0.5, 0.1),
                   metavar=("PEP", "MHC", "TORSION", "PLDDT"))
    p.add_argument("--groove-aware", action="store_true",
                   help="dock in-pocket peptides, expel out-of-pocket ones to a "
                        "shell (containment), learn pLDDT on all (loss-only; "
                        "resume-safe — no new parameters)")
    p.add_argument("--lambda-contain", type=float, default=0.5)
    p.add_argument("--groove", type=float, nargs=4, default=(8.0, 1.5, 8.0, 18.0),
                   metavar=("TAU_MID", "S", "TAU_OUT", "TAU_FAR"),
                   help="in/out boundary, sigmoid width, containment band (Å)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints_model2")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-eval-every", type=int, default=5,
                   help="every N epochs, DDIM-sample val complexes and report "
                        "peptide/MHC Cα-RMSD (comparable to model_1); 0=off")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    distributed, rank, local_rank, world = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)        # different diffusion noise per rank
    device = (f"cuda:{local_rank}" if distributed
              else args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lambdas = tuple(args.lambdas)

    train_loader, val_loader, train_sampler = M.build_loaders(
        args.h5_dir, args.scheme, args.fold, args.bs, args.num_workers,
        dummy=args.dummy, rank=rank, world_size=world)

    core = M.MHCDiff(T=args.timesteps, mhc_scale=args.mhc_scale, h_dim=args.hidden,
                     n_layers=args.layers, k=args.k, use_cross=not args.no_cross,
                     device=device, groove_aware=args.groove_aware,
                     lambda_contain=args.lambda_contain, groove=tuple(args.groove))
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=args.amp)

    start_epoch, best = 1, float("inf")
    if args.resume and Path(args.resume).exists():
        try:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            core.load_state_dict(ck["model"])
            # reject NaN/Inf weights (e.g. a checkpoint written after a divergence) —
            # better to fall back than to resume a poisoned model.
            if not all(torch.isfinite(p).all() for p in core.parameters()):
                raise ValueError("resumed weights contain NaN/Inf")
            # optimizer / scheduler / scaler are OPTIONAL: a model-only checkpoint
            # (e.g. best.pt, used to recover from a NaN run) reinitialises them.
            have_opt = all(k in ck for k in ("opt", "sched", "scaler"))
            if have_opt:
                opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
                scaler.load_state_dict(ck["scaler"])
            start_epoch, best = ck.get("epoch", 0) + 1, ck.get("best", float("inf"))
            if is_main:
                print(f"[m2] resumed at epoch {start_epoch} "
                      f"({'full state' if have_opt else 'MODEL ONLY -> fresh optimizer'}"
                      f", best {best:.3f})")
        except Exception as e:
            if is_main:
                print(f"[m2] WARNING bad checkpoint ({e}); starting fresh")

    # DDP wraps AFTER resume so the loaded weights are broadcast to all ranks.
    net = DDP(core, device_ids=[local_rank], find_unused_parameters=True) \
        if distributed else core

    run = args.run_name or f"mhcdiff_{args.scheme}_fold{args.fold}"
    run_dir = Path(args.ckpt_dir) / run
    cfg = {**vars(args), "lambdas": lambdas, "device": str(device), "world": world,
           "params": sum(p.numel() for p in core.parameters()),
           "n_train": len(train_loader.dataset), "created": time.ctime()}
    csv = run_dir / "metrics.csv"
    csv_cols = list(_COLS)
    if is_main:                                # only rank 0 writes config/logs/ckpts
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        csv_cols = _csv_cols(csv)
        print(f"[m2] run={run} device={device} world={world} "
              f"train={len(train_loader.dataset)} params={cfg['params']} "
              f"T={args.timesteps} k={args.k} layers={args.layers} "
              f"cross={not args.no_cross} lambdas={lambdas}")

    dev_type = "cuda"
    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)     # reshuffle shards each epoch
        net.train()
        t0 = time.perf_counter()
        for i, data in enumerate(train_loader, 1):
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev_type, enabled=args.amp):
                total, terms = net(data, lambdas)        # DDP forward -> compute_loss
            # non-finite guard: a single bad batch must NOT poison the weights (grad
            # clipping does not help — it propagates NaN). All ranks must agree to skip
            # together, else the DDP backward all-reduce deadlocks.
            bad = torch.tensor([0.0 if torch.isfinite(total) else 1.0], device=device)
            if distributed:
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
            if bad.item() > 0:
                opt.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[m2] epoch {epoch} step {i}: non-finite loss -> skipped")
                continue
            scaler.scale(total).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if is_main and args.log_every and i % args.log_every == 0:
                rate = i / (time.perf_counter() - t0)
                lr = opt.param_groups[0]["lr"]
                print(f"[m2] epoch {epoch} step {i}/{steps_per_epoch} | "
                      f"total {terms['total']:.3f} pep {terms['pep_coord']:.3f} "
                      f"mhc {terms['mhc_coord']:.3f} plddt {terms['plddt']:.3f} "
                      f"pep_plddt {terms['pep_plddt']:.3f} | lr {lr:.2e} "
                      f"| {rate:.1f} it/s")
                _csv_row(csv, csv_cols, wall_time=time.time(), split="train", epoch=epoch,
                         step=i, lr=lr, **terms)
        # evaluate + checkpoint on rank 0 only; other ranks wait
        if is_main:
            ev = evaluate(core, val_loader, device, lambdas)
            # periodic STRUCTURAL eval (the real, model_1-comparable number)
            if args.sample_eval_every and epoch % args.sample_eval_every == 0:
                ev.update(sample_rmsd(core, val_loader, device))
            _csv_row(csv, csv_cols, wall_time=time.time(), split="val", epoch=epoch, **ev)
            print(f"[m2] epoch {epoch:3d} | val total {ev['total']:.3f} "
                  f"pep {ev['pep_coord']:.3f} mhc {ev['mhc_coord']:.3f} "
                  f"plddt {ev['plddt']:.3f} pep_plddt {ev['pep_plddt']:.3f}"
                  + (f" | SAMPLED pep-RMSD {ev['pep_ca_rmsd']:.2f}Å "
                     f"mhc-RMSD {ev['mhc_ca_rmsd']:.2f}Å (n={ev['n']})"
                     if 'pep_ca_rmsd' in ev else ""))
            state = {"epoch": epoch, "model": core.state_dict(),
                     "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "best": best, "config": cfg}
            tmp = run_dir / "last.pt.tmp"
            torch.save(state, tmp); tmp.replace(run_dir / "last.pt")
            if ev["total"] < best:
                best = ev["total"]
                torch.save({"epoch": epoch, "model": core.state_dict(), "val": ev,
                            "config": cfg}, run_dir / "best.pt")
                print(f"[m2]   new best val total {best:.3f} -> best.pt")
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
