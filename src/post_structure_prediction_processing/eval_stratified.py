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


def _superpose_on(p, t, align, evalm) -> float:
    """RMSD over ``evalm`` points after Kabsch-aligning on the ``align`` points.
    Inlined (not imported from model_2/train.py) because ``pyg_data`` puts
    ``src/model`` on sys.path first, so ``import train`` would grab model_1's."""
    pa, ta = p[align], t[align]
    mp, mt = pa.mean(0), ta.mean(0)
    u, _, vt = torch.linalg.svd((pa - mp).T @ (ta - mt))
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    rot = vt.T @ torch.diag(torch.tensor([1., 1., d], device=p.device)) @ u.T
    return float((((p[evalm] - mp) @ rot.T - (t[evalm] - mt)) ** 2).sum(-1).mean().sqrt())


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
    # --recycles overrides the eval recycle count (None = model's configured value).
    # Use it to A/B whether recycling buys anything: 0 recycles bypasses the recycler
    # entirely (same forward cost as a non-recycling model), so if 0≈N, drop it.
    er = args.recycles if args.recycles is not None else (cfg.get("recycles", 0) or None)
    print(f"  [model_1] eval recycles = {er}")
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
                    # ground-truth peptide->nearest-MHC (teacher coords)
                    "pep_nndist_med": _median_nearest(batch["teacher_ca"][b][pm],
                                                       batch["teacher_ca"][b][am]),
                    # PREDICTED peptide->nearest-MHC (model's own coords): does the
                    # model put the peptide where THIS target wants it, or always in
                    # the canonical groove?
                    "pred_nndist_med": _median_nearest(ca[b][pm], ca[b][am]),
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
                              n_steps=args.n_steps, clip=args.clip,
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
                    "pep_nndist_med": _median_nearest(tt[pe], tt[~pe]),   # GT
                    "pred_nndist_med": _median_nearest(pp[pe], pp[~pe]),  # predicted
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


def groove_report_plot(df, setting, out_dir: Path, model_tag: str):
    """Does the model place the peptide where THIS target wants it, or always in the
    canonical groove? Compare the PREDICTED peptide->nearest-MHC distance against the
    GROUND-TRUTH distance. Mean-pose collapse => predicted is a narrow band
    (low std, slope~0 vs GT) regardless of where GT actually is."""
    import numpy as np
    gt = df["pep_nndist_med"].to_numpy(float)
    pr = df["pred_nndist_med"].to_numpy(float)
    # robust window so diverged samples (huge pred distance) don't dominate stats
    ok = np.isfinite(pr) & np.isfinite(gt) & (pr < 30) & (gt < 30)
    gtk, prk = gt[ok], pr[ok]
    corr = float(np.corrcoef(gtk, prk)[0, 1]) if ok.sum() > 2 else float("nan")
    slope = float(np.polyfit(gtk, prk, 1)[0]) if ok.sum() > 2 else float("nan")
    std_ratio = float(prk.std() / (gtk.std() + 1e-9))
    print(f"\n  --- groove placement (predicted vs GT peptide->nearest-MHC) ---")
    print(f"    GT   distance: mean {gtk.mean():.2f}  std {gtk.std():.2f} Å")
    print(f"    pred distance: mean {prk.mean():.2f}  std {prk.std():.2f} Å"
          f"   ({(~ok).sum()} diverged/clipped of {len(df)})")
    print(f"    corr(pred, GT) = {corr:+.3f}   (1 = tracks target, 0 = ignores it)")
    print(f"    regression slope pred~GT = {slope:+.3f}   (1 = adapts, 0 = mean pose)")
    print(f"    std(pred)/std(GT) = {std_ratio:.3f}   (<<1 = collapsed to a fixed pose)")
    verdict = ("LIKELY MEAN-POSE COLLAPSE" if (slope < 0.3 or std_ratio < 0.4)
               else "tracks target geometry" if slope > 0.6
               else "partially adapts")
    print(f"    => {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    lim = (min(gtk.min(), prk.min()) - 0.5, max(gtk.max(), prk.max()) + 0.5)
    ax[0].scatter(gtk, prk, s=8, alpha=0.25, color="#4C72B0")
    ax[0].plot(lim, lim, "k--", lw=1, label="y=x (perfect)")
    xs = np.linspace(*lim, 50)
    ax[0].plot(xs, np.polyval(np.polyfit(gtk, prk, 1), xs), color="#C44E52",
               lw=1.6, label=f"fit (slope {slope:.2f})")
    ax[0].axhline(prk.mean(), color="#999", ls=":", lw=1,
                  label=f"pred mean {prk.mean():.1f} Å")
    ax[0].set(xlim=lim, ylim=lim, xlabel="GROUND-TRUTH peptide nearest-MHC (Å)",
              ylabel="PREDICTED peptide nearest-MHC (Å)",
              title=f"placement vs target  (corr {corr:+.2f})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    ax[1].hist(gtk, bins=40, range=(3, 14), alpha=0.55, color="#55A868",
               density=True, label=f"ground truth (std {gtk.std():.2f})")
    ax[1].hist(prk, bins=40, range=(3, 14), alpha=0.55, color="#C44E52",
               density=True, label=f"predicted (std {prk.std():.2f})")
    ax[1].set(xlabel="peptide nearest-MHC (Å)", ylabel="density",
              title="distance distributions")
    ax[1].legend(fontsize=8)

    resid = prk - gtk
    ax[2].scatter(gtk, resid, s=8, alpha=0.25, color="#8172B3")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set(xlabel="GROUND-TRUTH nearest-MHC (Å)",
              ylabel="predicted − GT (Å)",
              title="residual (negative = pulled toward groove)")
    ax[2].grid(alpha=0.25)
    fig.suptitle(f"{model_tag} — groove placement ({setting}, n={int(ok.sum()):,}) "
                 f"| {verdict}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "groove_placement.png", dpi=140)
    plt.close(fig)
    print(f"    [plot] wrote {out_dir/'groove_placement.png'}")


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

    if "pred_nndist_med" in df.columns:
        groove_report_plot(df, setting, out_dir, model_tag)

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
    ap.add_argument("--clip", type=float, default=4.0,
                    help="model_2 only: DDIM x0-clip in normalized units (÷15 Å). "
                         "4≈±60 Å (loose); ~2.5 bounds reconstruction to physical "
                         "size and curbs sampling divergence.")
    ap.add_argument("--recycles", type=int, default=None,
                    help="model_1 only: override eval recycle count (0 bypasses the "
                         "recycler). Use to A/B whether recycling helps.")
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
