"""
model_multimer_1 training — three stages.

Data: TRAIN = old-store two_axis TRAIN ids (confidence-filtered in stage 1) + ALL
hasmig ids. VAL/TEST = old-store two_axis VAL/TEST ids ONLY (the HLA two-axis split
lives only on the old data; hasmig is a training-only, low-diversity augmentation).
Pick the split with --scheme/--fold. Validation is multi-GPU (sharded + all-reduced).

  stage 1 (structure, high-confidence): embedder + head-2 + trunk + projections trainable
    (SM & pLDDT head FROZEN), on confident/well-buried complexes only. pLDDT weight = 0
    for epoch 1, then --plddt-w. hasmig at full weight.
  stage 2 (broader data + confidence): SAME trainable set as stage 1 (SM stays FROZEN);
    broader data regime (Approach A filtered / B quality-weighted full); pLDDT weight
    --plddt-w. hasmig down-weighted (--hasmig-weight, default 0.1) in FAPE + pLDDT.
  stage 3 (confidence only): freeze EVERYTHING except the pLDDT projection (structure
    detached); pLDDT-only loss at --stage3-plddt-w. hasmig down-weighted.

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
import random
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
def store_has_sidechains(h5_dir) -> bool:
    """True iff the store's shards carry BOTH AF2 sidechain targets."""
    import glob, h5py
    shards = sorted(glob.glob(str(Path(h5_dir) / "*.h5")))
    if not shards:
        return False
    with h5py.File(shards[0], "r") as h:
        g = h[next(iter(h.keys()))]
        return ("teacher_atom14" in g) and ("teacher_chi" in g)


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


def quality_weight(p: float, b: float, w_min: float = 0.05) -> float:
    """Structure-quality weight  w_n = w_min + (1-w_min)·q_plddt(p)·q_burial(b).
    p,b in [0,1]; low-quality structures still contribute weakly (>= w_min) instead
    of being removed. Used by Approach B to weight the structural (FAPE) loss."""
    if   p >= 0.9: qp = 1.00
    elif p >= 0.8: qp = 0.85
    elif p >= 0.7: qp = 0.70
    elif p >= 0.6: qp = 0.50
    elif p >= 0.5: qp = 0.25
    else:          qp = 0.10
    qb = min(b / 0.65, 1.0)
    return w_min + (1.0 - w_min) * qp * qb


def _quality_map(args) -> dict:
    """id -> w_n for every store id, from the per-store score files. Old store:
    per_structure.csv (mean_peptide_plddt 0-100, burial_score). hasmig: index.csv
    (pep_mean_plddt 0-1, docking_score 0-1)."""
    m = {}
    if Path(args.data_exp_csv).exists():
        exp = pd.read_csv(args.data_exp_csv)
        for i, p, b in zip(exp["id"].astype(str),
                           exp["mean_peptide_plddt"].astype(float) / 100.0,
                           exp["burial_score"].astype(float)):
            m[i] = quality_weight(p, b, args.w_min)
    if args.hasmig_dir and (Path(args.hasmig_dir) / "index.csv").exists():
        idx = pd.read_csv(Path(args.hasmig_dir) / "index.csv")
        for i, p, b in zip(idx["id"].astype(str),
                           idx["pep_mean_plddt"].astype(float),
                           idx["docking_score"].astype(float)):
            m[i] = quality_weight(p, b, args.w_min)
    return m


class _WeightAnnotator(torch.utils.data.Dataset):
    """Annotate each example (by its `id`) with per-example loss weights:
      * sample_weight — source weight (scales FAPE + pLDDT CE); down-weights hasmig.
      * struct_weight — structure-quality w_n (scales FAPE ONLY, so the pLDDT CE still
        learns confidence on ALL structures). Missing id -> weight 1.0 in the loss."""

    def __init__(self, base, id2sw=None, id2stw=None):
        self.base = base
        self.id2sw = id2sw or {}
        self.id2stw = id2stw or {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        ex = self.base[i]
        iid = ex.get("id")
        if self.id2sw:
            ex["sample_weight"] = float(self.id2sw.get(iid, 1.0))
        if self.id2stw:
            ex["struct_weight"] = float(self.id2stw.get(iid, 1.0))
        return ex


def build_datasets(args, filt: bool, hasmig_w: float = 1.0):
    """Train = OLD-store two_axis TRAIN ids (confidence-filtered iff `filt`) + ALL
    hasmig ids (tagged with `hasmig_w`). Val = OLD-store two_axis VAL ids ONLY
    (the HLA two-axis split lives only on the old data; hasmig is never validated
    on). Returns (train, val_all, val_matched):
      * val_all     — the FULL held-out val fold (unfiltered; real distribution).
      * val_matched — the subset of val that passes the SAME confidence filter as
        stage-1 train, so train vs val_matched is an apples-to-apples check that
        isolates overfitting from the train/val distribution shift (most of val is
        low-confidence and never seen in stage-1 training). None if no filter."""
    train_parts, val_parts, val_hi_parts = [], [], []
    hasmig_id_set = set()

    # ---- OLD store: HLA two-axis split (the only source of val/test) ----
    if (Path(args.h5_dir) / "index.csv").exists():
        splits = m1.read_split_ids(args.scheme, args.fold)          # base-id lists
        train_base, val_base = set(splits["train"]), set(splits["val"])
        conf_ids, o_shard = old_ids(args.h5_dir, args.data_exp_csv, args.burial_min,
                                    args.plddt_min, filt)           # confident (or all)
        o_tr = [i for i in conf_ids if m1.base_id(i) in train_base]
        if o_tr:
            train_parts.append(m1.H5DistillDataset(o_tr, o_shard, args.h5_dir))
        all_idx, all_shard = _store_ids(args.h5_dir)
        full_va = [i for i in all_idx["id"] if m1.base_id(i) in val_base]
        conf_va = [i for i in conf_ids if m1.base_id(i) in val_base] if filt else []
        if args.filter_val and conf_va:                             # Approach A: val = filtered
            val_parts.append(m1.H5DistillDataset(conf_va, o_shard, args.h5_dir))
        elif full_va:                                               # val = full (real dist)
            val_parts.append(m1.H5DistillDataset(full_va, all_shard, args.h5_dir))
            if filt and conf_va:                                    # + val_matched diagnostic
                val_hi_parts.append(m1.H5DistillDataset(conf_va, o_shard, args.h5_dir))

    # ---- hasmig store: ALL ids -> TRAIN ONLY ----
    if args.hasmig_dir and (Path(args.hasmig_dir) / "index.csv").exists():
        h_ids, h_shard = hasmig_ids(args.hasmig_dir, args.burial_min,
                                    args.plddt_min, filt)
        if h_ids:
            train_parts.append(m1.H5DistillDataset(h_ids, h_shard, args.hasmig_dir))
            hasmig_id_set = set(h_ids)

    train_ds = ConcatDataset(train_parts) if train_parts else None
    val_ds = ConcatDataset(val_parts) if val_parts else None
    val_hi_ds = ConcatDataset(val_hi_parts) if val_hi_parts else None

    # Per-example loss weights: source (hasmig down-weight) and/or structure-quality w_n
    # (Approach B). Applied IDENTICALLY to train AND val(+val_matched) so the reported
    # validation loss uses the exact same weighting as train and is directly comparable.
    # (val holds only old ids, so the hasmig source weight is a no-op there; the quality
    # weight does apply.) No-op when both maps are empty.
    id2sw = {i: hasmig_w for i in hasmig_id_set} if hasmig_w != 1.0 else {}
    id2stw = _quality_map(args) if args.struct_quality_weight else {}
    if id2sw or id2stw:
        def _wrap(d):
            return _WeightAnnotator(d, id2sw, id2stw) if d is not None else None
        train_ds, val_ds, val_hi_ds = _wrap(train_ds), _wrap(val_ds), _wrap(val_hi_ds)
    return train_ds, val_ds, val_hi_ds


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--h5-dir", default="data/processed/h5_store")
    p.add_argument("--hasmig-dir", default="data/processed/h5_store_hasmig")
    p.add_argument("--data-exp-csv", default="outputs/data_exploration/per_structure.csv")
    p.add_argument("--scheme", default="two_axis", choices=["two_axis", "hla_only"],
                   help="HLA split scheme for the OLD store's train/val/test partition")
    p.add_argument("--fold", type=int, default=1, help="CV fold (1..5) for val")
    p.add_argument("--hasmig-weight", type=float, default=0.1,
                   help="per-example loss weight for hasmig data in stages 2/3 "
                        "(low-diversity, high-quality -> down-weight to curb overfit); "
                        "stage 1 always uses 1.0")
    p.add_argument("--burial-min", type=float, default=0.65)
    p.add_argument("--plddt-min", type=float, default=0.70)      # 0-1 (x100 for old store)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--plddt-w", type=float, default=0.01)        # stage-1/2 pLDDT weight
    p.add_argument("--stage3-plddt-w", type=float, default=1.0)  # stage-3 pLDDT weight
    p.add_argument("--no-filter", action="store_true", help="disable the stage-1 filter")
    p.add_argument("--force-filter", action="store_true",
                   help="apply the confidence filter in ANY stage (Approach A stage 2)")
    p.add_argument("--filter-val", action="store_true",
                   help="validate on the filtered val subset only (Approach A)")
    p.add_argument("--struct-quality-weight", action="store_true",
                   help="weight the structural (FAPE) loss per structure by w_n "
                        "(Approach B); full dataset, no filter")
    p.add_argument("--w-min", type=float, default=0.05, help="w_n floor (Approach B)")
    p.add_argument("--max-train", type=int, default=0,
                   help="cap the train set to N random examples (smoke/convergence test)")
    p.add_argument("--subset-val-frac", type=float, default=0.2,
                   help="with --max-train: fraction held out for validation (disjoint)")
    p.add_argument("--cap-train", type=int, default=0,
                   help="cap the train set to N random examples but KEEP the real "
                        "two_axis val / val-matched sets. Unlike --max-train (which "
                        "REPLACES val with a random 20%% slice of the train pool, so val "
                        "shares alleles and peptides with train and reads optimistically), "
                        "this leaves validation on the held-out HLA fold — use it for a "
                        "short run whose val number is still an honest generalisation "
                        "measurement.")
    p.add_argument("--max-val", type=int, default=0,
                   help="cap val (and val-matched) to N random examples, seeded so the "
                        "SAME subset is used every epoch and across runs. 0 = full set. "
                        "The full val fold is ~30k structures, which costs more than the "
                        "training it follows in a short run.")
    p.add_argument("--allow-no-val", action="store_true",
                   help="proceed even if no old (two_axis) store is present -> NO "
                        "validation (hasmig-only training). Off by default on purpose.")
    p.add_argument("--peptide-weight", type=float, default=5.0)
    p.add_argument("--mhc-noise", type=float, default=0.1)
    p.add_argument("--unfreeze-sm-pct", type=float, default=0.0,
                   help="unfreeze the last N%% of the StructureModule (0 = keep frozen)")
    p.add_argument("--unfreeze-sm-at", type=int, default=0,
                   help="epoch at which to apply --unfreeze-sm-pct (rebuilds opt/sched)")
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--sidechains", action="store_true",
                   help="enable AF2 sidechain supervision (sc-FAPE + supervised chi). "
                        "Requires a store preprocessed with --sidechains; FAILS otherwise.")
    p.add_argument("--sc-fape-w", type=float, default=0.5)   # AF: fape.sidechain.weight
    p.add_argument("--bb-fape-w", type=float, default=0.5)   # AF: fape.backbone.weight
    p.add_argument("--chi-w", type=float, default=1.0)       # AF: supervised_chi.weight
    p.add_argument("--unweighted-sidechain-losses", action="store_true",
                   help="restore the OLD (buggy) side-chain reduction: sc-FAPE and chi "
                        "as plain means over every residue of every example, escaping "
                        "--peptide-weight, --hasmig-weight and --struct-quality-weight. "
                        "Those two terms are ~71%% of the total loss and ~95%% MHC, so "
                        "the default (weighted) is the fix; this flag exists only to "
                        "A/B against the old behaviour.")
    p.add_argument("--pep-frames", choices=["teacher", "identity"], default="identity",
                   help="'teacher' feeds the peptide's TRUE backbone frames to the trunk "
                        "-> LEAKS the ground-truth pose (what all current checkpoints "
                        "were trained with; kept as default for reproducibility). "
                        "'identity' withholds them (documented design) -> the pose must "
                        "actually be predicted. Use 'identity' for a correct retrain.")
    p.add_argument("--trunk-fp32", default="tri",
                   help="comma-list of trunk ops to run in fp32 ('' = all bf16). The N^2 "
                        "triangle products amplify bf16 rounding into spurious gradient "
                        "spikes (max |g| 967 in bf16 vs 29 with tri in fp32; the loss is "
                        "identical). This is what killed the 13_08 run.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fape-clamp", type=float, default=0.0,
                   help="clamp the backbone FAPE at this distance (A). DEFAULT OFF: a 10 A "
                        "clamp starves the gradient at init (the peptide starts >10 A off) "
                        "and the model does not learn -- measured: pep-RMSD 10.8 A clamped "
                        "vs 7.2 A unclamped; AF2's 0.9-probability clamp is no better "
                        "(10.1 A). Only enable once the model already fits (<10 A errors).")
    p.add_argument("--fape-clamp-prob", type=float, default=0.9,
                   help="fraction of steps on which the backbone FAPE is clamped (AF2 uses "
                        "~0.9). 1.0 = always clamp -> the gradient dies at init and the "
                        "model does not learn (measured). Lower = more long-range signal.")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="linear LR warmup before the cosine decay (0 = none)")
    p.add_argument("--grad-spike-factor", type=float, default=10.0,
                   help="reject an optimizer step whose gradient norm exceeds this "
                        "multiple of the RUNNING MEDIAN norm (0 = off). grad_clip bounds "
                        "a spike's magnitude but keeps its DIRECTION, so a garbage "
                        "gradient still takes a full-size wrong step and compounds; job "
                        "29377637 died that way (max |g| 1-2 -> 9 -> 44 -> 186 -> 5.5e14 "
                        "over 1k steps). The non-finite guard only fires once the weights "
                        "have ALREADY collapsed; this catches the precursor.")
    p.add_argument("--grad-spike-warmup", type=int, default=200,
                   help="steps of history before spike rejection activates")
    p.add_argument("--divergence-factor", type=float, default=8.0,
                   help="abort if the window loss exceeds this x the best for "
                        "3 consecutive windows (0 = never abort)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ckpt-dir", default="checkpoints_mm1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--train-metrics-every", type=int, default=0,
                   help="also compute pep-RMSD/pLDDT-corr on the TRAIN batch every N "
                        "steps (0=off; e.g. 50). Adds a small Kabsch/Spearman cost.")
    p.add_argument("--fresh-optim", action="store_true",
                   help="on --resume, load WEIGHTS ONLY (fresh optimizer + LR schedule "
                        "+ epoch 0) even if the checkpoint is the same stage — use to "
                        "CONTINUE a finished/cut-off run with a new longer schedule")
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

    # filter train iff stage 1 (default) or explicitly forced (Approach A); --no-filter wins
    filt = ((args.stage == 1) or args.force_filter) and not args.no_filter
    hasmig_w = 1.0 if args.stage == 1 else args.hasmig_weight   # down-weight in 2/3
    # NO SILENT NO-OP: validation comes ONLY from the OLD (two_axis) store. If --h5-dir
    # doesn't exist, a real run would train hasmig-only with NO validation and never say
    # so. Fail loudly (unless --max-train, which builds its own held-out split, or the
    # explicit --allow-no-val escape hatch).
    old_store_ok = (Path(args.h5_dir) / "index.csv").exists()
    if not old_store_ok and args.max_train == 0 and not args.allow_no_val:
        raise SystemExit(
            f"FATAL: old (two_axis) store not found at --h5-dir '{args.h5_dir}' "
            f"(no index.csv). Validation comes ONLY from it, so this run would train on "
            f"hasmig alone with NO validation. Point --h5-dir at the two_axis store "
            f"(preprocess it with --sidechains), or pass --allow-no-val to proceed anyway.")
    train_ds, val_ds, val_hi_ds = build_datasets(args, filt=filt, hasmig_w=hasmig_w)
    if args.max_train > 0:                     # small-subset smoke / convergence test
        from torch.utils.data import Subset
        idx = list(range(len(train_ds)))
        random.Random(args.seed).shuffle(idx)                  # RANDOM subset (not first-N)
        idx = idx[:min(args.max_train, len(idx))]
        # DISJOINT 80/20 split -- never train == val, so the reported val number is a
        # real held-out measurement rather than memorisation.
        n_val = max(1, int(round(len(idx) * args.subset_val_frac)))
        val_idx, tr_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
        assert not (set(val_idx) & set(tr_idx)), "subset train/val overlap"
        val_ds = Subset(train_ds, val_idx)
        train_ds = Subset(train_ds, tr_idx)
        val_hi_ds = None
    if args.cap_train > 0:
        if args.max_train > 0:
            raise SystemExit("--cap-train and --max-train are mutually exclusive: "
                             "--max-train builds its own val split from the train pool, "
                             "--cap-train keeps the real two_axis val fold.")
        from torch.utils.data import Subset
        idx = list(range(len(train_ds)))
        random.Random(args.seed).shuffle(idx)                  # RANDOM subset
        train_ds = Subset(train_ds, sorted(idx[:min(args.cap_train, len(idx))]))
    if args.max_val > 0:
        from torch.utils.data import Subset

        def _cap_val(ds, seed):
            if ds is None or len(ds) <= args.max_val:
                return ds
            j = list(range(len(ds)))
            random.Random(seed).shuffle(j)
            return Subset(ds, sorted(j[:args.max_val]))
        val_ds = _cap_val(val_ds, args.seed + 1)
        val_hi_ds = _cap_val(val_hi_ds, args.seed + 2)
    shard_n = len(train_ds) // world if world > 1 else len(train_ds)
    steps_per_epoch = max(1, (shard_n + args.bs - 1) // args.bs)
    total_steps = steps_per_epoch * args.epochs

    model = MM.MultimerModel(n_trunk=args.n_trunk, mhc_noise=args.mhc_noise,
                             device=device, pep_frames=args.pep_frames,
                             trunk_fp32=tuple(x for x in args.trunk_fp32.split(",") if x))
    model.set_stage(args.stage)
    # stage 1: structure ONLY (lambda_plddt = 0); 2: structure + pLDDT(0.01);
    # 3: pLDDT ONLY (all structure terms off, model frozen except plddt_proj).
    bb_w = args.bb_fape_w if args.sidechains else 1.0   # AF splits FAPE 0.5/0.5
    lam = {1: (bb_w, 0.0, 0.0),
           2: (bb_w, args.plddt_w, 0.0),
           3: (0.0, args.stage3_plddt_w, 0.0)}[args.stage]
    struct_on = args.stage in (1, 2)
    loss_mod = m1.DistillLoss(
        *lam, peptide_weight=args.peptide_weight,
        fape_clamp=(args.fape_clamp if args.fape_clamp > 0 else None),
        fape_clamp_prob=args.fape_clamp_prob,
        lambda_sc_fape=(args.sc_fape_w if (args.sidechains and struct_on) else 0.0),
        lambda_chi=(args.chi_w if (args.sidechains and struct_on) else 0.0),
        weight_sidechain_losses=not args.unweighted_sidechain_losses,
    ).to(device)

    def build_opt_sched(t_max):
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                weight_decay=1e-4)
        t_max = max(1, t_max)
        warm = min(int(args.warmup_steps), max(0, t_max - 1))
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max - warm))
        if warm <= 0:
            return opt, cos
        # linear warmup then cosine: a cold start straight at the peak LR is a classic
        # way to walk into a bad region early.
        wu = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [wu, cos], milestones=[warm])
        return opt, sch

    optimizer, scheduler = build_opt_sched(total_steps)
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
        sd = model.trainable_state()
        # REFUSE to persist a poisoned model. The 13_08 run wrote NaN weights over
        # last.pt every 1000 steps, destroying the healthy weights from ~23k steps and
        # making the whole run unrecoverable.
        if not all(torch.isfinite(v).all() for v in sd.values()):
            log(f"[mm1] !! REFUSING to save non-finite weights to {Path(path).name} "
                f"at gstep {gs} — keeping the last good checkpoint")
            return
        tmp = Path(str(path) + ".tmp")
        torch.save({"epoch": ep, "global_step": gs, "trainable": sd,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "stage": args.stage}, tmp)
        tmp.replace(path)

    start_epoch, global_step = 1, 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["trainable"], strict=False)     # only trainable saved
        if ck.get("stage") == args.stage and not args.fresh_optim:  # same stage -> continue
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = int(ck["epoch"]) + 1
            global_step = int(ck["epoch"]) * steps_per_epoch
            scheduler.last_epoch = global_step
        mode = "weights-only (fresh optim/sched)" if (
            args.fresh_optim or ck.get("stage") != args.stage) else "continue"
        log(f"[mm1] resumed {args.resume} (stage {ck.get('stage')} -> {args.stage}, "
            f"{mode}), start epoch {start_epoch}")

    # ---- NO SILENT NO-OPS: fail at startup, not after a day of training ----------
    if args.sidechains:
        # check EVERY store actually used (train_ds[0] would only probe the first one;
        # a hasmig store without targets would have slipped through silently).
        used = [d for d in (args.h5_dir, args.hasmig_dir)
                if d and (Path(d) / "index.csv").exists()]
        bad = [d for d in used if not store_has_sidechains(d)]
        if bad:
            raise SystemExit(
                f"FATAL: --sidechains given but these stores lack teacher_atom14/"
                f"teacher_chi: {bad}. Re-run preprocessing with --sidechains. "
                f"(Refusing to train a silently-disabled loss.)")
        log(f"[mm1] sidechain targets present in all {len(used)} store(s): {used}")
        if args.stage in (1, 2):
            assert loss_mod.l_sc_fape > 0 and loss_mod.l_chi > 0, \
                "sidechain loss weights are zero in a structure stage"
        log(f"[mm1] sidechains ON: sc_fape_w={loss_mod.l_sc_fape} chi_w={loss_mod.l_chi} "
            f"(chi={loss_mod.chi_weight}, angle_norm={loss_mod.angle_norm_weight})")
        log(f"[mm1] sidechain-loss weighting: "
            f"{'WEIGHTED (peptide/sample/struct apply)' if loss_mod.weight_sidechain_losses else 'UNWEIGHTED (old behaviour)'}")
    if args.pep_frames == "identity":
        MM._leak_check(dev=device)          # pre-flight: peptide pose cannot leak

    n_tr = sum(p.numel() for p in model.trainable_parameters())
    if args.pep_frames == "teacher":
        log("[mm1] " + "!" * 72)
        log("[mm1] !! WARNING: --pep-frames teacher feeds the peptide's TRUE backbone")
        log("[mm1] !! frames to the trunk. The ground-truth peptide pose LEAKS in and")
        log("[mm1] !! every peptide metric will be meaningless (~0.3 A instead of ~8 A).")
        log("[mm1] !! Use --pep-frames identity for a real model. See README.")
        log("[mm1] " + "!" * 72)
    log(f"[mm1] stage {args.stage} | trainable {n_tr:,} params | train={len(train_ds)} "
        f"val={len(val_ds) if val_ds is not None else 0} bs={args.bs} world={world} "
        f"steps/epoch={steps_per_epoch} "
        f"filter={filt} amp_bf16={args.amp}" + (f" | ckpt={run_dir}" if run_dir else ""))

    net = (torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False) if distributed else model)

    def validate(ds):
        """Multi-GPU validation: every rank evaluates a disjoint, equal-size shard of
        ``ds``, then the loss/metric terms are all-reduced (example-weighted) so every
        rank ends up with the same global means. ALL ranks must call this (the
        all-reduce is a collective). Single-GPU falls back to a plain pass."""
        if ds is None or not len(ds):
            return None
        if not distributed:
            loader = m1.make_dataloader(ds, args.bs, shuffle=False,
                                        num_workers=args.num_workers)
            return m1.evaluate(model, loader, loss_mod, device, num_recycles=None)
        loader = m1.make_epoch_loader(ds, args.bs, args.num_workers, args.seed,
                                      epoch=0, rank=rank, world_size=world)
        local, n = m1.evaluate(model, loader, loss_mod, device,
                               num_recycles=None, return_n=True)
        keys = sorted(local.keys())          # identical on all ranks (equal shards)
        buf = torch.tensor([local[k] * n for k in keys] + [float(n)],
                           device=device, dtype=torch.float64)
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        tot = float(buf[-1].item())
        return {k: float(buf[i].item()) / max(tot, 1.0) for i, k in enumerate(keys)}

    for epoch in range(start_epoch, args.epochs + 1):
        # optionally unfreeze a slice of the StructureModule partway through, then
        # rebuild opt+sched over the new trainable set (fresh cosine over the rest)
        # and re-wrap DDP so the newly-trainable params are reduced.
        if args.unfreeze_sm_pct > 0 and epoch == args.unfreeze_sm_at:
            n_unf, n_tot = model.unfreeze_sm_pct(args.unfreeze_sm_pct)
            optimizer, scheduler = build_opt_sched(
                steps_per_epoch * (args.epochs - epoch + 1))
            if distributed:
                net = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
            n_tr = sum(p.numel() for p in model.trainable_parameters())
            log(f"[mm1] epoch {epoch}: unfroze {n_unf:,}/{n_tot:,} SM params "
                f"({args.unfreeze_sm_pct}%); trainable now {n_tr:,}; rebuilt opt/sched")
        # stage 1 is structure-only for its whole duration (lambda_plddt == 0, and
        # plddt_proj is frozen), so there is no pLDDT ramp any more.
        train_loader = m1.make_epoch_loader(train_ds, args.bs, args.num_workers,
                                            args.seed, epoch, rank=rank, world_size=world)
        tr, global_step = m1.train_one_epoch(
            net, train_loader, loss_mod, optimizer, scheduler, device,
            None, args.grad_clip, log=(log if is_main else None),
            log_every=args.log_every, epoch=epoch, global_step=global_step, mlog=mlog,
            ckpt_every=args.ckpt_every, core=model, is_main=is_main, amp_dtype=amp_dtype,
            metrics_every=args.train_metrics_every,
            divergence_factor=args.divergence_factor,
            grad_spike_factor=args.grad_spike_factor,
            grad_spike_warmup=args.grad_spike_warmup,
            save_fn=((lambda gs, ep: save_ckpt(run_dir / "last.pt", gs, ep))
                     if (run_dir is not None and is_main) else None),
            # snapshot the BEST model so a later blow-up cannot cost us the good weights
            best_save_fn=((lambda gs, ep: save_ckpt(run_dir / "best.pt", gs, ep))
                          if (run_dir is not None and is_main) else None))
        ev = validate(val_ds)                # collective: every rank participates
        ev_hi = validate(val_hi_ds)          # confidence-matched subset (== train dist)
        if is_main:
            log(f"[mm1] epoch {epoch:3d} | train total {tr['total']:.3f} "
                f"fape {tr['fape']:.3f} plddt_ce {tr.get('plddt_ce', 0):.3f} "
                f"(lam_plddt={loss_mod.l_plddt})")
            if ev:
                mlog.log("val", epoch, global_step, None, ev)
                log(f"[mm1]   val total {ev['total']:.3f} pep-pLDDT-MAE "
                    f"{ev.get('pep_plddt_mae', 0):.2f} pep-RMSD {ev.get('pep_ca_rmsd', 0):.2f}")
            if ev_hi:                        # apples-to-apples with train (same filter)
                mlog.log("val_matched", epoch, global_step, None, ev_hi)
                log(f"[mm1]   val-matched total {ev_hi['total']:.3f} pep-pLDDT-MAE "
                    f"{ev_hi.get('pep_plddt_mae', 0):.2f} pep-RMSD "
                    f"{ev_hi.get('pep_ca_rmsd', 0):.2f}  (confidence-filtered val)")
            save_ckpt(run_dir / "last.pt", global_step, epoch)
    mlog.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
