"""
model_3 training entrypoint — fine-tune a thin slice of the truncated AF2 Evoformer
by distilling teacher structures (FAPE + pLDDT/PAE CE). Reuses model_1's loss, data
pipeline, per-epoch loop, DDP setup and metric logging; only the model differs
(``EvoDistillModel``) and the checkpoint stores just the trainable params.

  # local smoke on the 15-example dummy set, 1 GPU
  python src/model_3/train.py --dummy --epochs 2 --bs 1 --evo-layers 3 --trainable 10
  # real run (DDP via torchrun, see train_af3.sbatch)
  python src/model_3/train.py --h5-dir data/processed/h5_store --scheme two_axis \
      --fold 1 --epochs 12 --bs 1 --evo-layers 3 --trainable 10 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M3                                       # noqa: E402
m1 = M3.m1                                               # model_1 utils (shared infra)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evo-layers", type=int, default=3,
                   help="K: number of first AF2 Evoformer blocks to keep (of 48)")
    p.add_argument("--trainable", type=float, default=10.0,
                   help="%% of the LAST kept Evoformer block to unfreeze (rest frozen)")
    p.add_argument("--unfreeze-sm", type=float, default=0.0,
                   help="%% of the frozen structure module to ALSO unfreeze (last params)")
    p.add_argument("--unfreeze-plddt", type=float, default=0.0,
                   help="%% of the pLDDT head to ALSO unfreeze")
    p.add_argument("--unfreeze-pae", type=float, default=0.0,
                   help="%% of the PAE/TM head to ALSO unfreeze")
    p.add_argument("--scheme", default="two_axis", choices=["two_axis", "hla_only"])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--h5-dir", default=None)
    p.add_argument("--ckpt-dir", default="checkpoints_model3")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambdas", type=float, nargs=3, default=(1.0, 0.01, 0.01),
                   metavar=("FAPE", "PLDDT", "PAE"))
    p.add_argument("--peptide-weight", type=float, default=5.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    m1.set_seed(args.seed)
    distributed, rank, local_rank, world = m1.setup_distributed()
    is_main = rank == 0
    device = (f"cuda:{local_rank}" if distributed
              else "cuda" if torch.cuda.is_available() else "cpu")
    log = print if is_main else (lambda *a, **k: None)

    # data (reuse model_1 store / loaders)
    if args.dummy:
        train_ds = m1.build_dataset(args.scheme, args.fold, "train", dummy=True)
        val_ds = m1.build_dataset(args.scheme, args.fold, "val", dummy=True)
    else:
        train_ds = m1.build_h5_dataset(args.h5_dir, args.scheme, args.fold, "train")
        val_ds = m1.build_h5_dataset(args.h5_dir, args.scheme, args.fold, "val")
    val_loader = m1.make_dataloader(val_ds, args.bs, shuffle=False,
                                    num_workers=args.num_workers)
    shard_n = len(train_ds) // world if world > 1 else len(train_ds)
    steps_per_epoch = max(1, (shard_n + args.bs - 1) // args.bs)
    total_steps = steps_per_epoch * args.epochs

    model = M3.EvoDistillModel(evo_layers=args.evo_layers, trainable=args.trainable,
                               device=device, unfreeze_sm=args.unfreeze_sm,
                               unfreeze_plddt=args.unfreeze_plddt,
                               unfreeze_pae=args.unfreeze_pae)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    loss_mod = m1.DistillLoss(*args.lambdas, peptide_weight=args.peptide_weight).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    # AF2/OpenFold must run in bf16, NOT fp16 (fp16 overflows the Evoformer -> NaN).
    # bf16 needs no GradScaler (fp32 exponent range), so we keep one only (disabled)
    # for checkpoint compatibility and use plain backward.
    amp_dtype = torch.bfloat16 if args.amp else None
    scaler = torch.amp.GradScaler(device="cuda", enabled=False)

    run_dir = None
    config = {"evo_layers": args.evo_layers, "trainable_pct": args.trainable,
              "unfreeze_sm": args.unfreeze_sm, "unfreeze_plddt": args.unfreeze_plddt,
              "unfreeze_pae": args.unfreeze_pae,
              "unfrozen": getattr(model, "unfrozen", None), "scheme": args.scheme,
              "fold": args.fold, "bs": args.bs, "lr": args.lr,
              "lambdas": tuple(args.lambdas), "peptide_weight": args.peptide_weight,
              "epochs": args.epochs, "trainable_params": n_train, "world": world}
    if args.ckpt_dir is not None:
        run_name = args.run_name or f"af3_{args.scheme}_fold{args.fold}_K{args.evo_layers}"
        # distinct dir when heads/SM are also unfrozen -> parallel run won't collide
        if not args.run_name and (args.unfreeze_sm or args.unfreeze_plddt
                                  or args.unfreeze_pae):
            run_name += (f"_uf{args.unfreeze_sm:g}-{args.unfreeze_plddt:g}"
                         f"-{args.unfreeze_pae:g}")
        run_dir = Path(args.ckpt_dir) / run_name
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    mlog = m1.MetricLogger(run_dir if is_main else None, enable_tb=True)

    def save_ckpt(path, global_step, epoch, best):
        if run_dir is None or not is_main:
            return
        tmp = Path(str(path) + ".tmp")
        torch.save({"epoch": epoch, "global_step": global_step,
                    "trainable": model.trainable_state(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best": best,
                    "config": config}, tmp)
        tmp.replace(path)

    start_epoch, best, global_step = 1, float("inf"), 0
    if args.resume and Path(args.resume).exists():
        try:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ck["trainable"], strict=False)   # trainable params only
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            best = ck.get("best", float("inf"))
            start_epoch = int(ck["epoch"]) + 1            # epoch-boundary resume
            global_step = int(ck["epoch"]) * steps_per_epoch
            scheduler.last_epoch = global_step
            log(f"[m3] resumed at epoch {start_epoch} (best {best:.3f})")
        except Exception as e:
            log(f"[m3] WARNING bad checkpoint ({e}); starting fresh")

    log(f"[m3] K={args.evo_layers} trainable={args.trainable}% "
        f"({n_train:,} params) | train={len(train_ds)} val={len(val_ds)} "
        f"bs={args.bs} world={world} steps/epoch={steps_per_epoch} amp={args.amp}"
        + (f" | ckpt={run_dir}" if run_dir else ""))

    # only the last Evoformer block trains and it's fully used every step -> no unused
    # params, so find_unused_parameters=False (faster, no extra graph traversal).
    net = (torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False) if distributed else model)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = m1.make_epoch_loader(train_ds, args.bs, args.num_workers,
                                            args.seed, epoch, rank=rank,
                                            world_size=world)
        tr, global_step = m1.train_one_epoch(
            net, train_loader, loss_mod, optimizer, scheduler, device,
            None, args.grad_clip,                      # bf16: plain backward, no scaler
            log=(log if is_main else None), log_every=args.log_every, epoch=epoch,
            global_step=global_step, mlog=mlog, ckpt_every=args.ckpt_every,
            core=model, is_main=is_main, amp_dtype=amp_dtype,
            save_fn=((lambda gs, ep: save_ckpt(run_dir / "last.pt", gs, ep, best))
                     if (run_dir is not None and is_main) else None))
        if is_main:
            ev = m1.evaluate(model, val_loader, loss_mod, device,
                             max_batches=args.eval_batches, num_recycles=None)
            mlog.log("val", epoch, global_step, None, ev)
            log(f"[m3] epoch {epoch:3d} | train total {tr['total']:.3f} fape "
                f"{tr['fape']:.3f} | val total {ev['total']:.3f} "
                f"Cα-RMSD {ev['ca_rmsd']:.2f} pep-RMSD {ev['pep_ca_rmsd']:.2f}")
            if run_dir is not None:
                if ev["ca_rmsd"] < best:
                    best = ev["ca_rmsd"]
                    torch.save({"epoch": epoch, "trainable": model.trainable_state(),
                                "val": ev, "config": config}, run_dir / "best.pt")
                    log(f"[m3]   new best val Cα-RMSD {best:.3f} -> best.pt")
                save_ckpt(run_dir / "last.pt", global_step, epoch, best)
    mlog.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
