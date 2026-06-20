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
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M                                       # noqa: E402

_COLS = ["wall_time", "split", "epoch", "step", "lr", "total",
         "pep_coord", "mhc_coord", "torsion", "plddt"]


def _csv_init(p: Path):
    if not p.exists():
        p.write_text(",".join(_COLS) + "\n")


def _csv_row(p: Path, **kw):
    with open(p, "a") as f:
        f.write(",".join("" if kw.get(c) is None else f"{kw.get(c)}"
                         for c in _COLS) + "\n")


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
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints_model2")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lambdas = tuple(args.lambdas)

    train_loader, val_loader = M.build_loaders(
        args.h5_dir, args.scheme, args.fold, args.bs, args.num_workers,
        dummy=args.dummy)
    net = M.MHCDiff(T=args.timesteps, mhc_scale=args.mhc_scale, h_dim=args.hidden,
                    n_layers=args.layers, k=args.k, use_cross=not args.no_cross,
                    device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=args.amp)

    run = args.run_name or f"mhcdiff_{args.scheme}_fold{args.fold}"
    run_dir = Path(args.ckpt_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "lambdas": lambdas, "device": str(device),
           "params": sum(p.numel() for p in net.parameters()),
           "n_train": len(train_loader.dataset), "created": time.ctime()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    csv = run_dir / "metrics.csv"
    _csv_init(csv)

    start_epoch, best = 1, float("inf")
    if args.resume and Path(args.resume).exists():
        try:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            start_epoch, best = ck["epoch"] + 1, ck.get("best", float("inf"))
            print(f"[m2] resumed at epoch {start_epoch} (best {best:.3f})")
        except Exception as e:
            print(f"[m2] WARNING bad checkpoint ({e}); starting fresh")

    print(f"[m2] run={run} device={device} train={len(train_loader.dataset)} "
          f"params={cfg['params']} T={args.timesteps} k={args.k} "
          f"layers={args.layers} cross={not args.no_cross} lambdas={lambdas}")
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        t0 = time.perf_counter()
        for i, data in enumerate(train_loader, 1):
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev_type, enabled=args.amp):
                total, terms = net.compute_loss(data, lambdas)
            scaler.scale(total).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if args.log_every and i % args.log_every == 0:
                rate = i / (time.perf_counter() - t0)
                lr = opt.param_groups[0]["lr"]
                print(f"[m2] epoch {epoch} step {i}/{steps_per_epoch} | "
                      f"total {terms['total']:.3f} pep {terms['pep_coord']:.3f} "
                      f"mhc {terms['mhc_coord']:.3f} plddt {terms['plddt']:.2f} "
                      f"| lr {lr:.2e} | {rate:.1f} it/s")
                _csv_row(csv, wall_time=time.time(), split="train", epoch=epoch,
                         step=i, lr=lr, **terms)
        ev = evaluate(net, val_loader, device, lambdas)
        _csv_row(csv, wall_time=time.time(), split="val", epoch=epoch, **ev)
        print(f"[m2] epoch {epoch:3d} | val total {ev['total']:.3f} "
              f"pep {ev['pep_coord']:.3f} mhc {ev['mhc_coord']:.3f} "
              f"plddt {ev['plddt']:.2f}")
        state = {"epoch": epoch, "model": net.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                 "best": best, "config": cfg}
        tmp = run_dir / "last.pt.tmp"
        torch.save(state, tmp); tmp.replace(run_dir / "last.pt")
        if ev["total"] < best:
            best = ev["total"]
            torch.save({"epoch": epoch, "model": net.state_dict(), "val": ev,
                        "config": cfg}, run_dir / "best.pt")
            print(f"[m2]   new best val total {best:.3f} -> best.pt")


if __name__ == "__main__":
    main()
