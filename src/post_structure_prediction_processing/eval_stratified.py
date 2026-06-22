"""
Stratified structural evaluation: peptide Cα-RMSD vs teacher data quality.

Runs a trained model on the val split and, per example, records the predicted
peptide Cα-RMSD alongside two TEACHER-derived difficulty axes:
  * median peptide pLDDT          (teacher confidence on the peptide)
  * median peptide nearest-MHC Å  (how deep in the groove the peptide sits)
then stratifies RMSD by each axis. This answers: do we predict the
low-confidence / poorly-docked peptides worse? If yes, the ceiling is the data,
and filtering/re-predicting those examples is the lever (not more training).

Works for both models (``--model 1`` distill, ``--model 2`` MHC-Diff). For
model_2 the MHC is initialised from the template pool by default (the deployment
setting); use ``--mhc-init truth`` to control the MHC and isolate peptide docking.

Run ON THE CLUSTER (needs the H5 store + GPU + openfold/torch):
  # model_1 (distill)
  python src/post_structure_prediction_processing/eval_stratified.py --model 1 \
      --ckpt checkpoints/two_axis_fold1_variant7_pw5.0_rc3/last.pt \
      --h5-dir data/processed/h5_store --scheme two_axis --fold 1 \
      --max-graphs 1500 --out-dir outputs/eval_stratified/model1_v7

  # model_2 (MHC-Diff)
  python src/post_structure_prediction_processing/eval_stratified.py --model 2 \
      --ckpt checkpoints_model2/mhcdiff_two_axis_fold1/last.pt \
      --h5-dir data/processed/h5_store_sc --scheme two_axis --fold 1 \
      --mhc-init template --n-steps 25 --max-graphs 1500 \
      --out-dir outputs/eval_stratified/model2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]

# difficulty-axis bins
_PLDDT_BINS = [0, 40, 50, 60, 70, 100]
_PLDDT_LBL = ["<40", "40-50", "50-60", "60-70", "70+"]
_NND_BINS = [0, 6, 7, 8, 9, 100]
_NND_LBL = ["<6", "6-7", "7-8", "8-9", "9+"]


def _median_nearest(pep_ca: torch.Tensor, mhc_ca: torch.Tensor) -> float:
    d = torch.cdist(pep_ca, mhc_ca)                  # [n_pep, n_mhc]
    return float(d.min(dim=1).values.median())


# --------------------------------------------------------------------------- #
# model_1 (distillation)
# --------------------------------------------------------------------------- #
def eval_model1(args, device):
    sys.path.insert(0, str(_SRC / "model"))
    import utils as U
    cfg = json.loads((Path(args.ckpt).parent / "config.json").read_text())
    model = U.DistillModel(
        cfg["variant"], device=device, recycles=cfg.get("recycles", 0),
        unfreeze_sm=cfg.get("unfreeze_sm", 0.0),
        unfreeze_plddt=cfg.get("unfreeze_plddt", 0.0),
        unfreeze_pae=cfg.get("unfreeze_pae", 0.0),
        anchor_relpos=cfg.get("anchor_relpos", False)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ck["encoder"])
    if ck.get("frozen_trainable"):
        model.load_state_dict(ck["frozen_trainable"], strict=False)
    model.eval()

    ds = U.build_h5_dataset(args.h5_dir, args.scheme, args.fold, args.split)
    loader = U.make_dataloader(ds, args.bs, shuffle=False, num_workers=args.num_workers)
    er = cfg.get("recycles", 0) or None
    recs, seen = [], 0
    with torch.no_grad():
        for batch in loader:
            batch = U.move_batch(batch, device)
            ca, _, _ = model(batch, return_frames=False, num_recycles=er)
            sm = batch["seq_mask"]
            pep = U.peptide_mask_from_batch(sm, batch["segment_id"])
            for b in range(ca.shape[0]):
                m = sm[b].bool()
                pm = pep[b].bool() & m
                am = (~pep[b].bool()) & m
                if int(am.sum()) < 3 or int(pm.sum()) < 1:
                    continue
                rmsd = U._superpose_rmsd_on(ca[b], batch["teacher_ca"][b], am, pm)
                if rmsd is None or not torch.isfinite(rmsd):
                    continue
                tp = batch["teacher_plddt"][b][pm]
                recs.append({
                    "pep_ca_rmsd": float(rmsd),
                    "pep_plddt_med": float(tp.median()),
                    "pep_nndist_med": _median_nearest(batch["teacher_ca"][b][pm],
                                                       batch["teacher_ca"][b][am]),
                })
            seen += ca.shape[0]
            if args.max_graphs and seen >= args.max_graphs:
                break
    return recs, "own predicted MHC"


# --------------------------------------------------------------------------- #
# model_2 (MHC-Diff)
# --------------------------------------------------------------------------- #
def eval_model2(args, device):
    sys.path.insert(0, str(_SRC / "model_2"))
    import model as M
    import pyg_data as PD
    from torch_geometric.loader import DataLoader
    from train import _superpose_on                    # same-dir helper

    cfg = json.loads((Path(args.ckpt).parent / "config.json").read_text())
    net = M.MHCDiff(T=cfg["timesteps"], mhc_scale=cfg["mhc_scale"],
                    h_dim=cfg["hidden"], n_layers=cfg["layers"], k=cfg["k"],
                    use_cross=not cfg["no_cross"], device=device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()

    base = M.PD.m1.build_h5_dataset(args.h5_dir, args.scheme, args.fold, args.split)
    val = PD.H5GraphDataset(base)
    pool = PD.MHCTemplatePool(base) if args.mhc_init == "template" else None
    setting = {"template": "MHC from template pool", "truth": "MHC from truth",
               "noise": "MHC from noise"}[args.mhc_init]
    recs, seen = [], 0
    with torch.no_grad():
        for data in DataLoader(val, batch_size=args.bs):
            data = data.to(device)
            pred = net.sample(data, template_pool=pool, sampler="ddim",
                              n_steps=args.n_steps,
                              mhc_from_truth=(args.mhc_init == "truth"))
            for g in range(data.num_graphs):
                m = data.batch == g
                pp, tt, pe = pred[m], data.pos[m], data.pep[m].bool()
                if not torch.isfinite(pp).all() or int(pe.sum()) < 1:
                    continue
                rmsd = _superpose_on(pp, tt, ~pe, pe)
                tp = data.teacher_plddt[m][pe]
                recs.append({
                    "pep_ca_rmsd": float(rmsd),
                    "pep_plddt_med": float(tp.median()),
                    "pep_nndist_med": _median_nearest(tt[pe], tt[~pe]),
                })
            seen += data.num_graphs
            if args.max_graphs and seen >= args.max_graphs:
                break
    return recs, setting


# --------------------------------------------------------------------------- #
# stratify + report + plot
# --------------------------------------------------------------------------- #
def _strat(df, col, bins, labels):
    import pandas as pd
    cut = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby(cut, observed=False)["pep_ca_rmsd"]
    return g.agg(n="size", median="median", mean="mean",
                 p25=lambda s: s.quantile(.25),
                 p75=lambda s: s.quantile(.75)).reset_index()


def report_plot(recs, setting, out_dir: Path, model_tag: str):
    import pandas as pd
    df = pd.DataFrame(recs)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_example.csv", index=False)

    sp = df["pep_ca_rmsd"].corr(df["pep_plddt_med"])
    sn = df["pep_ca_rmsd"].corr(df["pep_nndist_med"])
    print(f"\n=== {model_tag} stratified eval ({len(df):,} examples; {setting}) ===")
    print(f"  overall peptide Cα-RMSD: median {df.pep_ca_rmsd.median():.2f} Å "
          f"(mean {df.pep_ca_rmsd.mean():.2f})")
    print(f"  corr(RMSD, peptide pLDDT)        = {sp:+.3f}  (expect negative)")
    print(f"  corr(RMSD, peptide nearest-MHC)  = {sn:+.3f}  (expect positive)")

    sP = _strat(df, "pep_plddt_med", _PLDDT_BINS, _PLDDT_LBL)
    sN = _strat(df, "pep_nndist_med", _NND_BINS, _NND_LBL)
    print("\n  by teacher peptide pLDDT:")
    for _, r in sP.iterrows():
        print(f"    {str(r['pep_plddt_med']):<7} n={int(r['n']):>5}  "
              f"median RMSD {r['median']:.2f} Å")
    print("  by peptide nearest-MHC distance:")
    for _, r in sN.iterrows():
        print(f"    {str(r['pep_nndist_med']):<7} n={int(r['n']):>5}  "
              f"median RMSD {r['median']:.2f} Å")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # boxplots by bin
    for axi, (s, lbls, xlab) in [
        (ax[0, 0], (sP, _PLDDT_LBL, "teacher peptide pLDDT")),
        (ax[0, 1], (sN, _NND_LBL, "peptide nearest-MHC (Å)"))]:
        x = range(len(s))
        axi.bar(x, s["median"], color="#4C72B0", alpha=0.85)
        axi.errorbar(x, s["median"], yerr=[s["median"] - s["p25"], s["p75"] - s["median"]],
                     fmt="none", ecolor="#222", capsize=3)
        for xi, (_, r) in zip(x, s.iterrows()):
            axi.text(xi, r["p75"] + 0.1, f"n={int(r['n'])}", ha="center", fontsize=7)
        axi.set_xticks(list(x))
        axi.set_xticklabels(lbls)
        axi.set(xlabel=xlab, ylabel="peptide Cα-RMSD (Å) [median, IQR]")
        axi.grid(axis="y", alpha=0.25)
    ax[0, 0].set_title("RMSD vs teacher confidence")
    ax[0, 1].set_title("RMSD vs groove depth")
    # scatters
    ax[1, 0].scatter(df["pep_plddt_med"], df["pep_ca_rmsd"], s=6, alpha=0.25, color="#4C72B0")
    ax[1, 0].set(xlabel="teacher peptide pLDDT", ylabel="peptide Cα-RMSD (Å)",
                 title=f"corr = {sp:+.2f}")
    ax[1, 1].scatter(df["pep_nndist_med"], df["pep_ca_rmsd"], s=6, alpha=0.25, color="#55A868")
    ax[1, 1].set(xlabel="peptide nearest-MHC (Å)", ylabel="peptide Cα-RMSD (Å)",
                 title=f"corr = {sn:+.2f}")
    for a in (ax[1, 0], ax[1, 1]):
        a.grid(alpha=0.25)
    fig.suptitle(f"{model_tag} — peptide RMSD stratified by teacher data quality "
                 f"({setting}, n={len(df):,})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "stratified.png", dpi=140)
    plt.close(fig)
    print(f"\n[plot] wrote {out_dir/'stratified.png'}")
    print(f"[csv]  wrote {out_dir/'per_example.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=int, choices=[1, 2], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--scheme", default="two_axis")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-graphs", type=int, default=1500,
                    help="cap examples for speed (0 = all val)")
    ap.add_argument("--mhc-init", choices=["template", "truth", "noise"],
                    default="template", help="model_2 only: MHC initialisation")
    ap.add_argument("--n-steps", type=int, default=25, help="model_2 DDIM steps")
    ap.add_argument("--out-dir", default="outputs/eval_stratified")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == 1:
        recs, setting = eval_model1(args, device)
        tag = f"model_1 ({Path(args.ckpt).parent.name})"
    else:
        recs, setting = eval_model2(args, device)
        tag = f"model_2 ({Path(args.ckpt).parent.name})"
    if not recs:
        print("[error] no examples evaluated")
        return
    report_plot(recs, setting, Path(args.out_dir), tag)


if __name__ == "__main__":
    main()
