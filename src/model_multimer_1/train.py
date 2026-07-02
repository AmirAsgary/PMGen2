"""
model_multimer_1 training — two stages.

  stage 1 (high-confidence): train the whole model (embedder + head-2 + trunk +
    projections; SM & pLDDT head FROZEN) on ONLY confident, well-buried complexes
    (burial >= --burial-min AND peptide pLDDT > --plddt-min), from BOTH the old store
    (scores from outputs/data_exploration/per_structure.csv) and the new hasmig store
    (scores in its index.csv). pLDDT loss weight = 0 for epoch 1, then --plddt-w.

  stage 2 (confidence only): load stage-1 weights, freeze EVERYTHING except the pLDDT
    projection, train on ALL structures with pLDDT-only loss (structure detached).

Reuses model_1's DistillLoss / train_one_epoch / DDP / metric logging. AF-multimer
runs in bf16 (fp16 overflows). forward returns (ca, plddt, zeros_pae, frames), so
DistillLoss with lambda_pae=0 works unchanged.

  # stage 1
  torchrun ... src/model_multimer_1/train.py --stage 1 \
      --h5-dir data/processed/h5_store --hasmig-dir data/processed/h5_store_hasmig \
      --data-exp-csv outputs/data_exploration/per_structure.csv --amp
  # stage 2 (resume the structure from stage 1)
  torchrun ... src/model_multimer_1/train.py --stage 2 \
      --resume checkpoints_mm1/mm1_stage1/last.pt ... --amp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                                    # noqa: E402
m1 = MM.m1                                                            # model_1 utils


# --------------------------------------------------------------------------- #
# confident-subset id selection (stage 1) / all ids (stage 2)
# --------------------------------------------------------------------------- #
def _store_ids(h5_dir):
    idx = pd.read_csv(Path(h5_dir) / "index.csv", dtype={"id": str, "shard": str})
    return idx, dict(zip(idx["id"], idx["shard"]))


def old_ids(h5_dir, data_exp_csv, burial_min, plddt_min, filt):
    if not (Path(h5_dir) / "index.csv").exists():        # old store optional (e.g. local)
        return [], {}
    idx, id2shard = _store_ids(h5_dir)
    if not filt:
        return idx["id"].tolist(), id2shard
    exp = pd.read_csv(data_exp_csv)                       # id, burial_score, mean_peptide_plddt(0-100)
    keep = set(exp.loc[(exp["burial_score"] >= burial_min)
                       & (exp["mean_peptide_plddt"] > plddt_min * 100.0), "id"])
    return [i for i in idx["id"] if i in keep], id2shard


def hasmig_ids(h5_dir, burial_min, plddt_min, filt):
    idx, id2shard = _store_ids(h5_dir)
    if filt:
        d = idx["docking_score"].astype(float)            # burial (0-1)
        p = idx["pep_mean_plddt"].astype(float)           # confidence (0-1)
        idx = idx[(d >= burial_min) & (p > plddt_min)]
    return idx["id"].tolist(), id2shard


def _split(ids, val_frac, seed):
    import random
    r = random.Random(seed)
    ids = list(ids)
    r.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac)) if ids else 0
    return ids[n_val:], ids[:n_val]


def build_datasets(args, filt: bool):
    """(train_concat, val_concat) over old + hasmig stores (filtered iff `filt`)."""
    train_parts, val_parts = [], []
    o_ids, o_shard = old_ids(args.h5_dir, args.data_exp_csv, args.burial_min,
                             args.plddt_min, filt)
    o_tr, o_va = _split(o_ids, args.val_frac, args.seed)
    if o_tr:
        train_parts.append(m1.H5DistillDataset(o_tr, o_shard, args.h5_dir))
    if o_va:
        val_parts.append(m1.H5DistillDataset(o_va, o_shard, args.h5_dir))
    if args.hasmig_dir and (Path(args.hasmig_dir) / "index.csv").exists():
        h_ids, h_shard = hasmig_ids(args.hasmig_dir, args.burial_min,
                                    args.plddt_min, filt)
        h_tr, h_va = _split(h_ids, args.val_frac, args.seed)
        train_parts.append(m1.H5DistillDataset(h_tr, h_shard, args.hasmig_dir))
        if h_va:
            val_parts.append(m1.H5DistillDataset(h_va, h_shard, args.hasmig_dir))
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--h5-dir", default="data/processed/h5_store")
    p.add_argument("--hasmig-dir", default="data/processed/h5_store_hasmig")
    p.add_argument("--data-exp-csv", default="outputs/data_exploration/per_structure.csv")
    p.add_argument("--burial-min", type=float, default=0.65)
    p.add_argument("--plddt-min", type=float, default=0.70)      # 0-1 (x100 for old store)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--plddt-w", type=float, default=0.01)        # stage-1/2 pLDDT weight
    p.add_argument("--stage3-plddt-w", type=float, default=1.0)  # stage-3 pLDDT weight
    p.add_argument("--no-filter", action="store_true", help="disable the stage-1 filter")
    p.add_argument("--max-train", type=int, default=0, help="cap #train examples (overfit)")
    p.add_argument("--peptide-weight", type=float, default=5.0)
    p.add_argument("--mhc-noise", type=float, default=0.5)
    p.add_argument("--n-trunk", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ckpt-dir", default="checkpoints_mm1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1000)
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
    amp_dtype = torch.bfloat16 if args.amp else None            # AF-multimer needs bf16

    filt = (args.stage == 1) and not args.no_filter            # only stage 1 filters
    train_ds, val_ds = build_datasets(args, filt=filt)
    if args.max_train > 0:                                      # overfit / subset
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(args.max_train, len(train_ds)))))
        val_ds = train_ds                                      # watch it fit the same few
    shard_n = len(train_ds) // world if world > 1 else len(train_ds)
    steps_per_epoch = max(1, (shard_n + args.bs - 1) // args.bs)
    total_steps = steps_per_epoch * args.epochs

    model = MM.MultimerModel(n_trunk=args.n_trunk, mhc_noise=args.mhc_noise,
                             device=device)
    model.set_stage(args.stage)
    # stage 1: FAPE only (pLDDT set per-epoch below); 2: FAPE + pLDDT; 3: pLDDT only
    lam = {1: (1.0, 0.0, 0.0), 2: (1.0, args.plddt_w, 0.0),
           3: (0.0, args.stage3_plddt_w, 0.0)}[args.stage]
    loss_mod = m1.DistillLoss(*lam, peptide_weight=args.peptide_weight).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=False)

    run_dir = None
    if args.ckpt_dir:
        run_name = args.run_name or f"mm1_stage{args.stage}"
        run_dir = Path(args.ckpt_dir) / run_name
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2,
                                                            default=str))
    mlog = m1.MetricLogger(run_dir if is_main else None, enable_tb=True)

    def save_ckpt(path, gs, ep):
        if run_dir is None or not is_main:
            return
        tmp = Path(str(path) + ".tmp")
        torch.save({"epoch": ep, "global_step": gs, "trainable": model.trainable_state(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "stage": args.stage}, tmp)
        tmp.replace(path)

    start_epoch, global_step = 1, 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["trainable"], strict=False)     # only trainable saved
        if ck.get("stage") == args.stage:                        # same stage -> continue
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = int(ck["epoch"]) + 1
            global_step = int(ck["epoch"]) * steps_per_epoch
            scheduler.last_epoch = global_step
        log(f"[mm1] resumed {args.resume} (stage {ck.get('stage')} -> {args.stage}), "
            f"start epoch {start_epoch}")

    n_tr = sum(p.numel() for p in model.trainable_parameters())
    log(f"[mm1] stage {args.stage} | trainable {n_tr:,} params | train={len(train_ds)} "
        f"val={len(val_ds)} bs={args.bs} world={world} steps/epoch={steps_per_epoch} "
        f"filter={filt} amp_bf16={args.amp}" + (f" | ckpt={run_dir}" if run_dir else ""))

    net = (torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True) if distributed else model)
    val_loader = m1.make_dataloader(val_ds, args.bs, shuffle=False,
                                    num_workers=args.num_workers) if len(val_ds) else None

    for epoch in range(start_epoch, args.epochs + 1):
        # pLDDT-weight schedule (stage 1): 0 for epoch 1, then --plddt-w
        if args.stage == 1:
            loss_mod.l_plddt = 0.0 if epoch == 1 else args.plddt_w
        train_loader = m1.make_epoch_loader(train_ds, args.bs, args.num_workers,
                                            args.seed, epoch, rank=rank, world_size=world)
        tr, global_step = m1.train_one_epoch(
            net, train_loader, loss_mod, optimizer, scheduler, device,
            None, args.grad_clip, log=(log if is_main else None),
            log_every=args.log_every, epoch=epoch, global_step=global_step, mlog=mlog,
            ckpt_every=args.ckpt_every, core=model, is_main=is_main, amp_dtype=amp_dtype,
            save_fn=((lambda gs, ep: save_ckpt(run_dir / "last.pt", gs, ep))
                     if (run_dir is not None and is_main) else None))
        if is_main:
            log(f"[mm1] epoch {epoch:3d} | train total {tr['total']:.3f} "
                f"fape {tr['fape']:.3f} plddt_ce {tr.get('plddt_ce', 0):.3f} "
                f"(lam_plddt={loss_mod.l_plddt})")
            if val_loader is not None:
                ev = m1.evaluate(model, val_loader, loss_mod, device, num_recycles=None)
                mlog.log("val", epoch, global_step, None, ev)
                log(f"[mm1]   val total {ev['total']:.3f} pep-pLDDT-MAE "
                    f"{ev.get('pep_plddt_mae', 0):.2f} pep-RMSD {ev.get('pep_ca_rmsd', 0):.2f}")
            save_ckpt(run_dir / "last.pt", global_step, epoch)
    mlog.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
