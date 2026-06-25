"""
model4 / plddt_model — stage-1 pretraining entrypoint (AFDB monomer).

Objectives: pLDDT 50-bin CE (primary) + masked-LM CE (aux) + anchor/contact BCE
(aux). The AF pLDDT head is initialised from params/alphafold/plddt_af2.pt and
frozen for --freeze-plddt-epochs epochs, then unfrozen. bf16 autocast, AdamW +
cosine LR, DDP via torchrun, rank-0 logging/checkpointing (mirrors model_3).

  # local smoke (1 process, few proteins)
  python src/model4/train.py --h5 data/afdb/plddt_dataset.h5 \
      --domains-tsv data/afdb/final_domains_mapped_clean.tsv \
      --limit 200 --epochs 2 --bs 2 --num-workers 0
  # real run: see src/model4/pretrain_afdb.sbatch (torchrun, A100)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M                                            # noqa: E402
from data import AFDBMonomerDataset, worker_init_fn          # noqa: E402

AF_PLDDT_DEFAULT = "params/alphafold/plddt_af2.pt"


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--domains-tsv", default=None)
    p.add_argument("--crop-len", type=int, default=256)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--c-s", type=int, default=384)
    p.add_argument("--c-z", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--both-tri-mul", action="store_true")
    p.add_argument("--mlm-frac", type=float, default=0.15)
    p.add_argument("--anchor-reveal-prob", type=float, default=0.2)
    p.add_argument("--reveal-frac-max", type=float, default=0.5)
    p.add_argument("--pe-reveal-frac", type=float, default=0.5)
    p.add_argument("--neg-reveal-frac", type=float, default=0.02)
    p.add_argument("--anchor-seq-sep", type=int, default=6)
    p.add_argument("--neg-ratio", type=float, default=4.0, help="anchor-loss neg:pos")
    p.add_argument("--freeze-plddt-epochs", type=int, default=1,
                   help="epochs the AF pLDDT head stays fully frozen before unfreezing")
    p.add_argument("--gradual-unfreeze", action=argparse.BooleanOptionalAction, default=True,
                   help="after the freeze, unfreeze the pLDDT head one layer/epoch "
                        "(output->input); --no-gradual-unfreeze unfreezes it all at once")
    p.add_argument("--pair-fp32", action="store_true",
                   help="force the pair stack to fp32 (only needed under fp16 amp; "
                        "default bf16 keeps it in autocast for tensor-core speed)")
    p.add_argument("--lambdas", type=float, nargs=3, default=[1.0, 0.5, 0.5],
                   help="[plddt, mlm, anchor]")
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--min-len", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--af-plddt-path", default=AF_PLDDT_DEFAULT)
    p.add_argument("--no-af-plddt", action="store_true")
    p.add_argument("--ckpt-dir", default="checkpoints_model4")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=None)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world
    return False, 0, 0, 1


def make_loader(ds, bs, num_workers, epoch, distributed, rank, world, seed, shuffle):
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=shuffle, seed=seed, drop_last=False)
        sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(ds) if shuffle else None
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=False,
                      worker_init_fn=worker_init_fn,
                      persistent_workers=bool(num_workers))


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #
def plddt_ce(logits, plddt, mask, no_bins=50):
    b = torch.clamp((plddt * no_bins).long(), max=no_bins - 1)
    ce = F.cross_entropy(logits.reshape(-1, no_bins), b.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def mlm_ce(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100)


def anchor_bce(logits, target, loss_mask, neg_ratio):
    pos = (target > 0.5) & (loss_mask > 0.5)
    neg = (target < 0.5) & (loss_mask > 0.5)
    n_pos = int(pos.sum().item())
    if n_pos == 0:
        keep = neg & (torch.rand_like(target) < 0.001)
    else:
        p_keep = min(1.0, neg_ratio * n_pos / max(1, int(neg.sum().item())))
        keep = neg & (torch.rand_like(target) < p_keep)
    sel = pos | keep
    if sel.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[sel], target[sel])


def _pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-8)
    return (x * y).sum() / denom


def _spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    return _pearson(rx, ry)


@torch.no_grad()
def batch_metrics(out, batch):
    m = batch["residue_mask"] > 0.5
    pred = M.predicted_plddt(out["plddt_logits"])[m] / 100.0
    true = batch["plddt"][m]
    metr = {"plddt_mae": float((pred - true).abs().mean() * 100) if m.any() else 0.0,
            "plddt_pear": float(_pearson(pred, true)) if m.sum() > 2 else 0.0,
            "plddt_spear": float(_spearman(pred, true)) if m.sum() > 2 else 0.0}
    lbl = batch["mlm_labels"]; mm = lbl != -100
    if mm.any():
        acc = (out["mlm_logits"].argmax(-1)[mm] == lbl[mm]).float().mean()
        metr["mlm_acc"] = float(acc)
    am = batch["anchor_loss_mask"] > 0.5
    if am.any():
        ap = (torch.sigmoid(out["anchor_logits"])[am] > 0.5).float()
        at = batch["anchor_target"][am]
        metr["anchor_acc"] = float((ap == at).float().mean())
    return metr


def compute_loss(out, batch, lambdas, neg_ratio):
    lp = plddt_ce(out["plddt_logits"], batch["plddt"], batch["residue_mask"])
    lm = mlm_ce(out["mlm_logits"], batch["mlm_labels"])
    la = anchor_bce(out["anchor_logits"], batch["anchor_target"],
                    batch["anchor_loss_mask"], neg_ratio)
    total = lambdas[0] * lp + lambdas[1] * lm + lambdas[2] * la
    return total, {"plddt_ce": float(lp), "mlm_ce": float(lm), "anchor_bce": float(la)}


def l1_penalty(params):
    return sum(p.abs().sum() for p in params if p.requires_grad)


# --------------------------------------------------------------------------- #
def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main(argv=None):
    args = parse_args(argv)
    distributed, rank, local_rank, world = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    # TF32 for any residual fp32 matmuls (cheap global speedup on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_name = args.run_name or f"afdb_b{args.blocks}_cs{args.c_s}"
    run_dir = Path(args.ckpt_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # data
    common = dict(h5_path=args.h5, domains_tsv=args.domains_tsv, crop_len=args.crop_len,
                  anchor_seq_sep=args.anchor_seq_sep, reveal_prob=args.anchor_reveal_prob,
                  reveal_frac_max=args.reveal_frac_max, pe_reveal_frac=args.pe_reveal_frac,
                  neg_reveal_frac=args.neg_reveal_frac, mlm_frac=args.mlm_frac,
                  val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
                  min_len=args.min_len, max_offset=32)
    train_ds = AFDBMonomerDataset(split="train", limit=args.limit, **common)
    val_ds = AFDBMonomerDataset(split="val",
                                limit=(args.limit if args.limit else None), **common)
    test_ds = AFDBMonomerDataset(split="test",
                                 limit=(args.limit if args.limit else None), **common)
    if is_main:
        print(f"split 80/10/10 -> train={len(train_ds)} val={len(val_ds)} "
              f"test={len(test_ds)} | device={device} ddp={distributed} world={world}")

    # model
    model = M.PlddtModel(c_s=args.c_s, c_z=args.c_z, n_blocks=args.blocks,
                         dropout=args.dropout, both_tri_mul=args.both_tri_mul,
                         pair_fp32=args.pair_fp32).to(device)
    if not args.no_af_plddt and Path(args.af_plddt_path).exists():
        model.load_af_plddt(args.af_plddt_path)
        if is_main:
            print(f"loaded AF pLDDT head from {args.af_plddt_path}")

    net = model
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // (args.bs * world))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if is_main:
        summary = M.summarize_model(model)
        print(summary)
        print(f"schedule: {args.epochs} epochs x {steps_per_epoch} steps/epoch "
              f"= {total_steps} total optimizer steps (bs={args.bs} x world={world})")
        (run_dir / "architecture.txt").write_text(
            summary + f"\n\n{args.epochs} epochs x {steps_per_epoch} steps = "
            f"{total_steps} total steps\n")

    start_epoch, global_step, best = 0, 0, float("inf")
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck.get("global_step", 0))
        best = float(ck.get("best", best))
        if is_main:
            print(f"resumed from {args.resume} @ epoch {start_epoch}")

    log_path = run_dir / "metrics.csv"
    log_cols = ["wall", "split", "epoch", "step", "lr", "total", "plddt_ce", "mlm_ce",
                "anchor_bce", "plddt_mae", "plddt_spear", "mlm_acc", "anchor_acc"]
    if is_main and not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_cols)

    def log_row(split, epoch, step, lr, losses, metr):
        if not is_main:
            return
        row = [round(time.time(), 1), split, epoch, step, lr,
               losses.get("total", 0.0), losses.get("plddt_ce", 0.0),
               losses.get("mlm_ce", 0.0), losses.get("anchor_bce", 0.0),
               metr.get("plddt_mae", 0.0), metr.get("plddt_spear", 0.0),
               metr.get("mlm_acc", 0.0), metr.get("anchor_acc", 0.0)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([round(x, 4) if isinstance(x, float) else x for x in row])

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if args.amp and device.type == "cuda" else torch.autocast("cpu", enabled=False))

    n_head_groups = len(model.plddt_head_groups())
    for epoch in range(start_epoch, args.epochs):
        # pLDDT-head freeze schedule: fully frozen for the first
        # --freeze-plddt-epochs, then gradually unfrozen one layer/epoch
        # (output->input), or all-at-once if --no-gradual-unfreeze.
        if epoch < args.freeze_plddt_epochs:
            n_unfreeze = 0
        elif args.gradual_unfreeze:
            n_unfreeze = min(epoch - args.freeze_plddt_epochs + 1, n_head_groups)
        else:
            n_unfreeze = n_head_groups
        unfrozen = model.set_plddt_unfreeze(n_unfreeze)
        if is_main:
            print(f"epoch {epoch}/{args.epochs}: pLDDT head unfrozen layers = "
                  f"{unfrozen if unfrozen else 'NONE (frozen)'}")
        net.train()
        loader = make_loader(train_ds, args.bs, args.num_workers, epoch,
                             distributed, rank, world, args.seed, shuffle=True)
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = net(batch)
                total, losses = compute_loss(out, batch, args.lambdas, args.neg_ratio)
            if args.l1 > 0:
                total = total + args.l1 * l1_penalty(model.parameters())
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if is_main and global_step % args.log_every == 0:
                losses["total"] = float(total)
                metr = batch_metrics(out, batch)
                its = (it + 1) / (time.time() - t0)
                print(f"e{epoch}/{args.epochs} s{global_step}/{total_steps} "
                      f"loss={float(total):.3f} plddt={losses['plddt_ce']:.3f} "
                      f"mlm={losses['mlm_ce']:.3f} anch={losses['anchor_bce']:.3f} "
                      f"spear={metr.get('plddt_spear',0):.3f} {its:.2f}it/s")
                log_row("train", epoch, global_step, scheduler.get_last_lr()[0], losses, metr)
            if is_main and global_step % args.ckpt_every == 0:
                save_ckpt(run_dir / "last.pt", model, optimizer, scheduler,
                          epoch, global_step, best, vars(args))

        # validation (rank 0)
        if is_main:
            vmetr, vloss = evaluate(net, val_ds, args, device, amp_ctx)
            print(f"[val] e{epoch} plddt_ce={vloss['plddt_ce']:.3f} "
                  f"spear={vmetr.get('plddt_spear',0):.3f} mae={vmetr.get('plddt_mae',0):.2f} "
                  f"mlm_acc={vmetr.get('mlm_acc',0):.3f} anch_acc={vmetr.get('anchor_acc',0):.3f}")
            log_row("val", epoch, global_step, scheduler.get_last_lr()[0], vloss, vmetr)
            save_ckpt(run_dir / "last.pt", model, optimizer, scheduler,
                      epoch, global_step, best, vars(args))
            if vloss["plddt_ce"] < best:
                best = vloss["plddt_ce"]
                save_ckpt(run_dir / "best.pt", model, optimizer, scheduler,
                          epoch, global_step, best, vars(args))
        if distributed:
            torch.distributed.barrier()

    if distributed:
        torch.distributed.destroy_process_group()


@torch.no_grad()
def evaluate(net, ds, args, device, amp_ctx):
    net.eval()
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    agg_l, agg_m, n = {}, {}, 0
    for it, batch in enumerate(loader):
        if args.eval_batches and it >= args.eval_batches:
            break
        batch = to_device(batch, device)
        with amp_ctx:
            out = net(batch)
            _, losses = compute_loss(out, batch, args.lambdas, args.neg_ratio)
        metr = batch_metrics(out, batch)
        for k, v in losses.items():
            agg_l[k] = agg_l.get(k, 0.0) + v
        for k, v in metr.items():
            agg_m[k] = agg_m.get(k, 0.0) + v
        n += 1
    net.train()
    n = max(1, n)
    return {k: v / n for k, v in agg_m.items()}, {k: v / n for k, v in agg_l.items()}


def save_ckpt(path, model, optimizer, scheduler, epoch, step, best, config):
    torch.save({"epoch": epoch, "global_step": step, "best": best,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": config}, path)


if __name__ == "__main__":
    main()
