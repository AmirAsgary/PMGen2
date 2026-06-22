"""
Cross-model overview for PMGen-v2: model_2 (MHC-Diff) training curves + a
normalised best-model comparison (table + plot) spanning model_1 and model_2.

Why a separate script from ``plot_training.py``: model_2's ``metrics.csv`` has a
different schema (ε-prediction ``pep_coord``/``mhc_coord``/``torsion`` instead of
FAPE/CE), and the headline comparison must be on a metric the two models share.
That metric is the **structural Cα-RMSD** (peptide and MHC) — the per-run loss
values are NOT comparable across models (different objectives) nor across model_1
runs trained with different λ, so we compare on RMSD and only report each run's
own (raw) loss alongside for provenance.

Outputs (into --out-dir):
  model2_train.png, model2_val.png       -- MHC-Diff loss curves
  best_models_comparison.png + .csv      -- best model_1 variants vs model_2

Run (local; reads the copied tmp/ run dirs):
  python src/visualization/plot_overview.py \
      --model1-runs tmp/checkpoints \
      --model2-csv  tmp/checkpoints_mhcdiff_06_21_2026/metrics.csv \
      --out-dir outputs/visualization
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

# model_1 reference λ to reconstruct a comparable total (see plot_training.py)
REF_LAMBDAS = (1.0, 0.1, 0.1)


# --------------------------------------------------------------------------- #
# model_2 (MHC-Diff) curves
# --------------------------------------------------------------------------- #
_M2_TRAIN = [("total", "total loss"), ("pep_coord", "peptide ε-loss"),
             ("mhc_coord", "MHC ε-loss"), ("torsion", "torsion loss")]
_M2_VAL = [("total", "total loss"), ("pep_coord", "peptide ε-loss"),
           ("mhc_coord", "MHC ε-loss"), ("torsion", "torsion loss"),
           ("pep_plddt", "peptide pLDDT MSE"), ("pep_ca_rmsd", "peptide Cα-RMSD (Å)"),
           ("mhc_ca_rmsd", "MHC Cα-RMSD (Å)")]


def _smooth(y, frac):
    if frac <= 0 or len(y) < 3:
        return y
    win = max(1, int(len(y) * frac))
    return np.convolve(y, np.ones(win) / win, mode="same")


def _epoch_float(sub: pd.DataFrame) -> pd.Series:
    """Continuous x in fractional epochs. The logged ``epoch`` is an integer and
    ``step`` resets every epoch, so MANY train rows share one epoch — plotting vs
    ``epoch`` alone stacks them on a single x (vertical zig-zag). Spread them within
    [epoch, epoch+1) by the per-epoch step fraction."""
    if "step" not in sub.columns or sub["step"].notna().sum() == 0:
        return sub["epoch"].astype(float)
    smax = float(sub["step"].max())
    if smax <= 0:
        return sub["epoch"].astype(float)
    return sub["epoch"].astype(float) + sub["step"].astype(float) / (smax + 1.0)


def plot_model2(csv: Path, out_dir: Path, smooth: float = 0.04):
    df = pd.read_csv(csv)
    for split, panels, sm, fname in [
        ("train", _M2_TRAIN, smooth, "model2_train.png"),
        ("val", _M2_VAL, 0.0, "model2_val.png")]:
        sub = df[df["split"] == split].copy()
        sub["epoch_float"] = _epoch_float(sub)
        panels = [(c, lbl) for c, lbl in panels
                  if c in sub.columns and sub[c].notna().any()]
        n = len(panels)
        ncol = 2
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), squeeze=False)
        for i, (c, lbl) in enumerate(panels):
            ax = axes[i // ncol][i % ncol]
            s = sub[["epoch_float", c]].dropna().sort_values("epoch_float")
            xs, ys = s["epoch_float"].to_numpy(float), s[c].to_numpy(float)
            if split == "train":                          # faint raw + bold trend
                ax.plot(xs, ys, color="#3b6", lw=0.5, alpha=0.25)
                ax.plot(xs, _smooth(ys, sm), color="#176", lw=1.8)
                # zoom y to the trend, not the per-step noise spikes
                lo, hi = np.percentile(ys, [1, 97])
                ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.05 * (hi - lo))
            else:
                ax.plot(xs, ys, color="#3b6", lw=1.6)
            if c in ("pep_ca_rmsd", "mhc_ca_rmsd"):       # sampled: mark points
                ax.plot(xs, ys, "o", color="#176", ms=4)
                best = np.nanmin(ys)
                ax.annotate(f"best {best:.2f} Å", xy=(0.97, 0.93),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=8, color="#176")
            ax.set(xlabel="epoch", ylabel=lbl)
            ax.grid(alpha=0.25)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"MHC-Diff (model_2) — {split}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / fname, dpi=140)
        plt.close(fig)
        print(f"wrote {out_dir/fname}  ({n} panels, {split})")


# --------------------------------------------------------------------------- #
# best-model comparison
# --------------------------------------------------------------------------- #
def _best_val_row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    v = df[(df["split"] == "val") & df[key].notna()]
    return None if v.empty else v.loc[v[key].astype(float).idxmin()]


def collect_model1(root: Path) -> List[Dict]:
    rows = []
    for cfg_path in sorted(root.glob("*/config.json")):
        d = cfg_path.parent
        csv = d / "metrics.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if "pep_ca_rmsd" not in df.columns:
            continue
        best = _best_val_row(df, "pep_ca_rmsd")
        if best is None:
            continue
        lam = cfg.get("lambdas", REF_LAMBDAS)
        rows.append({
            "model": "model_1", "run": d.name,
            "variant": cfg.get("variant"), "fold": cfg.get("fold"),
            "pep_ca_rmsd": float(best["pep_ca_rmsd"]),
            "mhc_or_global_ca_rmsd": float(best.get("ca_rmsd", np.nan)),
            "pep_plddt_mae": float(best.get("pep_plddt_mae", np.nan)),
            "best_epoch": int(best["epoch"]),
            "lambdas": ",".join(f"{x:g}" for x in lam),
            "plddt_w_struct": cfg.get("plddt_weight_struct", False),
            "rmsd_setting": "own MHC",
        })
    return rows


def collect_model2(csv: Path) -> List[Dict]:
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    cfg = {}
    cfgp = csv.parent / "config.json"
    if cfgp.exists():
        cfg = json.loads(cfgp.read_text())
    best = _best_val_row(df, "pep_ca_rmsd")
    if best is None:
        return []
    return [{
        "model": "model_2", "run": cfg.get("run_name", csv.parent.name),
        "variant": None, "fold": cfg.get("fold"),
        "pep_ca_rmsd": float(best["pep_ca_rmsd"]),
        "mhc_or_global_ca_rmsd": float(best.get("mhc_ca_rmsd", np.nan)),
        "pep_plddt_mae": np.nan,                         # different scale (MSE/100)
        "best_epoch": int(best["epoch"]),
        "lambdas": ",".join(f"{x:g}" for x in cfg.get("lambdas", [])),
        "plddt_w_struct": False,
        "rmsd_setting": "MHC from truth",               # IMPORTANT caveat
    }]


def comparison(model1_root: Path, model2_csv: Path, out_dir: Path,
               max_rmsd: float = 10.0):
    rows = collect_model1(model1_root) + collect_model2(model2_csv)
    if not rows:
        print("[compare] no runs found")
        return
    tbl = pd.DataFrame(rows).sort_values("pep_ca_rmsd").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "best_models_comparison.csv"
    tbl.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}  ({len(tbl)} runs)")

    # console table
    print("\n=== BEST MODELS (ranked by peptide Cα-RMSD) ===")
    print("NB: folds are different CV splits => different val sets; absolute RMSD is\n"
          "    only comparable WITHIN a fold. model_2 lives on fold 1.")
    print(f"{'run':<34} {'fold':>4} {'pep_RMSD':>9} {'MHC/glob':>9} "
          f"{'pep_plddtMAE':>12} {'epoch':>5}  setting")
    for _, r in tbl.iterrows():
        print(f"{r['run']:<34} {str(r['fold']):>4} {r['pep_ca_rmsd']:>8.2f}A "
              f"{r['mhc_or_global_ca_rmsd']:>8.2f}A {r['pep_plddt_mae']:>12.2f} "
              f"{r['best_epoch']:>5}  {r['rmsd_setting']}")

    # bar plot — drop failed runs (sanity), keep them only in the CSV
    plot_tbl = tbl[tbl["pep_ca_rmsd"] <= max_rmsd].reset_index(drop=True)
    n_drop = len(tbl) - len(plot_tbl)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(plot_tbl) + 3), 5.4))
    # colour by model; hatch model_1 fold-1 runs (the fold that has model_2) so the
    # only truly like-for-like bars stand out.
    colors, hatches = [], []
    for _, r in plot_tbl.iterrows():
        colors.append("#C44E52" if r["model"] == "model_2" else "#4C72B0")
        hatches.append("//" if (r["model"] == "model_1" and r["fold"] == 1) else "")
    bars = ax.bar(range(len(plot_tbl)), plot_tbl["pep_ca_rmsd"], color=colors)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_xticks(range(len(plot_tbl)))
    ax.set_xticklabels(plot_tbl["run"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("best peptide Cα-RMSD (Å)")
    ax.set_title("Best-model comparison — peptide Cα-RMSD (lower better)")
    for b, (_, r) in zip(bars, plot_tbl.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{r['pep_ca_rmsd']:.2f}", ha="center", va="bottom", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4C72B0", label="model_1 (own MHC)"),
                       Patch(facecolor="#4C72B0", hatch="//", label="model_1 fold 1"),
                       Patch(facecolor="#C44E52", label="model_2 fold 1 (MHC from truth)")],
              fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    note = ("Caveats: (1) different folds = different val sets — compare only within a "
            "fold;\n(2) model_2 RMSD uses the TRUE MHC (easier) — not 1:1 with "
            "model_1's own-MHC RMSD.")
    if n_drop:
        note += f"\n({n_drop} failed run(s) with RMSD>{max_rmsd:g} Å hidden; see CSV.)"
    ax.annotate(note, xy=(0.5, 0.98), xycoords="axes fraction", ha="center",
                va="top", fontsize=7.5, color="#555")
    fig.tight_layout()
    fig.savefig(out_dir / "best_models_comparison.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_dir/'best_models_comparison.png'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model1-runs", default="tmp/checkpoints")
    ap.add_argument("--model2-csv",
                    default="tmp/checkpoints_mhcdiff_06_21_2026/metrics.csv")
    ap.add_argument("--out-dir", default="outputs/visualization")
    ap.add_argument("--smooth", type=float, default=0.03)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    plot_model2(Path(args.model2_csv), out, args.smooth)
    comparison(Path(args.model1_runs), Path(args.model2_csv), out)


if __name__ == "__main__":
    main()
