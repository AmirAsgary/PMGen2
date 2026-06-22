"""
Visualize PMGen-v2 training history from per-run ``metrics.csv`` (+ ``config.json``).

Compares runs (variants by default) side by side across several metrics in one
figure. Handles BOTH metrics.csv schemas (older runs without peptide columns,
newer runs with ``pep_*``) — missing columns are simply skipped.

Run-dir layout (one per run, as written by train.py / train_grid.sbatch):
    <ckpt_root>/<scheme>_fold<F>_variant<V>[_pw<W>]/{metrics.csv, config.json}

The figure is a grid of panels: one ROW per metric, one COLUMN per ``--col``
value (default: peptide_weight, so w=1 vs w=5 sit side by side), and one colored
line per ``--hue`` value (default: variant). When several runs fall in the same
panel/colour (e.g. the 5 CV folds), they are averaged into a mean line with a
±std band (disable with --no-aggregate to draw every run).

Examples
--------
# the 7 variants compared across the default metrics (mean over folds), w=1 vs w=5
python src/visualization/plot_training.py --runs tmp/checkpoints -o training.png

# only two_axis fold 1, raw train-loss curves (no fold averaging)
python src/visualization/plot_training.py --runs tmp/checkpoints \
    --scheme two_axis --fold 1 --split train --no-aggregate -o fold1_train.png

# focus on the peptide metrics, columns = fold
python src/visualization/plot_training.py --runs tmp/checkpoints --split val \
    --metrics pep_ca_rmsd pep_pae_mae ca_rmsd --col fold -o peptide.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")                       # headless / cluster-safe
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

# Reference loss weights used to reconstruct a *comparable* total across runs that
# were trained with different λ (we cut λ_plddt/λ_pae from 0.1 to 0.01 mid-project).
# The logged `fape`/`plddt_ce`/`pae_ce` are RAW (un-weighted), so re-mixing them with
# one fixed λ gives a `total_norm` curve that is continuous across that change.
REF_LAMBDAS = (1.0, 0.1, 0.1)               # (FAPE, pLDDT, PAE)

# metric -> human label (everything else falls back to the column name)
_LABELS = {
    "total": "total loss (as-trained)",
    "total_norm": f"total loss (norm λ={REF_LAMBDAS})",
    "fape": "FAPE", "plddt_ce": "pLDDT CE",
    "pae_ce": "PAE CE", "pep_fape": "peptide FAPE",
    "pep_plddt_ce": "peptide pLDDT CE", "pep_pae_ce": "peptide PAE CE",
    "ca_rmsd": "Cα-RMSD (Å)", "pep_ca_rmsd": "peptide Cα-RMSD (Å)",
    "plddt_spearman": "pLDDT Spearman", "pae_mae": "PAE-MAE (Å)",
    "pep_plddt_mae": "peptide pLDDT-MAE", "pep_pae_mae": "peptide PAE-MAE (Å)",
    "lr": "learning rate", "it_per_s": "it/s",
}
# sensible default metric sets per split (only those present are kept). Use the
# λ-independent `total_norm` / raw components so curves stay comparable across the
# loss-weight change.
_DEFAULT_TRAIN = ["total_norm", "fape", "plddt_ce", "pae_ce",
                  "pep_fape", "pep_plddt_ce", "pep_pae_ce"]
_DEFAULT_VAL = ["ca_rmsd", "pep_ca_rmsd", "pae_mae", "pep_pae_mae",
                "plddt_spearman", "total_norm"]
# metrics where LOWER is better (annotate the best run)
_LOWER_BETTER = {"total", "total_norm", "fape", "plddt_ce", "pae_ce", "pep_fape",
                 "pep_plddt_ce", "pep_pae_ce", "ca_rmsd", "pep_ca_rmsd",
                 "pae_mae", "pep_plddt_mae", "pep_pae_mae"}

_RUN_RE = re.compile(r"^(?P<scheme>.+?)_fold(?P<fold>\d+)_variant(?P<variant>\d+)"
                     r"(?:_pw(?P<pw>[\d.]+))?(?:_rc(?P<rc>\d+))?"
                     r"(?:_uf(?P<uf>[\d.\-]+))?$")


def _parse_run_name(name: str) -> Dict[str, object]:
    """Metadata from the dir name (scheme/fold/variant + pw/recycles/unfreeze)."""
    m = _RUN_RE.match(name)
    if not m:
        return {"scheme": "?", "fold": -1, "variant": -1, "peptide_weight": 1.0,
                "recycles": 0, "config": name}
    pw = float(m["pw"]) if m["pw"] else 1.0
    rc = int(m["rc"]) if m["rc"] else 0
    # compact config label for faceting: e.g. base / pw5 / pw5_rc3
    cfg = "base" if (pw == 1.0 and rc == 0) else (
        (f"pw{pw:g}" if pw != 1.0 else "") + (f"_rc{rc}" if rc else "")).strip("_")
    return {"scheme": m["scheme"], "fold": int(m["fold"]),
            "variant": int(m["variant"]), "peptide_weight": pw,
            "recycles": rc, "config": cfg}


def load_run(run_dir: Path) -> Optional[Tuple[Dict[str, object], pd.DataFrame]]:
    """Return (meta, dataframe) for one run dir, or None if it has no metrics."""
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
    except Exception:
        return None
    if df.empty or "split" not in df.columns:
        return None

    meta = _parse_run_name(run_dir.name)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            for k in ("scheme", "fold", "variant", "peptide_weight"):
                if k in cfg and cfg[k] is not None:
                    meta[k] = cfg[k]
            meta["bs"] = cfg.get("bs")
            meta["n_train"] = cfg.get("n_train")
        except Exception:
            pass
    meta["run"] = run_dir.name

    # λ-normalised total: re-mix the RAW components with one fixed λ so the curve is
    # continuous across runs trained with different loss weights. (`total` itself is
    # left as-trained for reference.) record the run's own λ for the report.
    lf, lp, la = REF_LAMBDAS
    if {"fape", "plddt_ce", "pae_ce"}.issubset(df.columns):
        df["total_norm"] = lf * df["fape"] + lp * df["plddt_ce"] + la * df["pae_ce"]
    if cfg_path.exists():
        try:                                    # provenance string (scalar, not tuple)
            lam = json.loads(cfg_path.read_text()).get("lambdas", None)
            meta["lambdas"] = ",".join(f"{x:g}" for x in lam) if lam else ""
        except Exception:
            pass

    # x in fractional epochs, derived from the ROW ORDER within each epoch (per split)
    # rather than global_step. This is immune to a global_step basis change across a
    # resume — e.g. single-GPU (83615 steps/epoch) -> DDP (41808/epoch) resets/rescales
    # the counter — which would otherwise fold the post-resume curve back on itself.
    df = df.reset_index(drop=True)
    grp = df.groupby(["split", "epoch"], sort=False)
    ordn = grp.cumcount()
    cnt = grp["epoch"].transform("size").clip(lower=1)
    df["epoch_float"] = (df["epoch"].astype(float) - 1.0) + (ordn + 1) / cnt
    for col, val in meta.items():
        df[col] = val
    return meta, df


def discover(roots: Sequence[Path]) -> pd.DataFrame:
    """Walk the given roots for run dirs (any dir containing metrics.csv)."""
    frames = []
    for root in roots:
        root = Path(root)
        candidates = ([root] if (root / "metrics.csv").exists()
                      else sorted(p.parent for p in root.glob("*/metrics.csv")))
        for d in candidates:
            loaded = load_run(d)
            if loaded is not None:
                frames.append(loaded[1])
    if not frames:
        raise SystemExit(f"no runs with metrics.csv found under {list(roots)}")
    return pd.concat(frames, ignore_index=True)


def _apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.scheme:
        df = df[df["scheme"].isin(args.scheme)]
    if args.fold:
        df = df[df["fold"].isin(args.fold)]
    if args.variant:
        df = df[df["variant"].isin(args.variant)]
    if args.weight is not None:
        df = df[df["peptide_weight"].isin(args.weight)]
    return df


def _xy(sub: pd.DataFrame, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray]:
    s = sub[[xcol, ycol]].dropna().sort_values(xcol)
    return s[xcol].to_numpy(float), s[ycol].to_numpy(float)


def _aggregate(runs: List[pd.DataFrame], xcol: str, ycol: str, n: int = 200):
    """Interpolate each run onto a shared x-grid, return (x, mean, std)."""
    series = [_xy(r, xcol, ycol) for r in runs]
    series = [(x, y) for x, y in series if len(x) >= 2]
    if not series:
        return None
    lo = min(x.min() for x, _ in series)
    hi = max(x.max() for x, _ in series)
    grid = np.linspace(lo, hi, n)
    stacked = []
    for x, y in series:
        yi = np.interp(grid, x, y)
        yi[(grid < x.min()) | (grid > x.max())] = np.nan   # no extrapolation
        stacked.append(yi)
    arr = np.vstack(stacked)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    return grid, mean, std


def _smooth(y: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0 or len(y) < 3:
        return y
    win = max(1, int(len(y) * frac))
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def plot(df: pd.DataFrame, metrics: List[str], hue: str, col: Optional[str],
         xcol: str, aggregate: bool, band: bool, smooth: float,
         title: str, out: Path) -> None:
    col_vals = sorted(df[col].dropna().unique()) if col else [None]
    ncols = len(col_vals)
    nrows = len(metrics)
    hue_vals = sorted(df[hue].dropna().unique())
    cmap = plt.get_cmap("tab10" if len(hue_vals) <= 10 else "tab20")
    colors = {h: cmap(i % cmap.N) for i, h in enumerate(hue_vals)}

    header = 1.1                              # inches reserved for title + legend
    fig_h = 2.6 * nrows + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 4.2 * ncols, fig_h),
                             squeeze=False, sharex=True)
    for r, metric in enumerate(metrics):
        for c, cv in enumerate(col_vals):
            ax = axes[r][c]
            cell = df if cv is None else df[df[col] == cv]
            best_txt = None
            for h in hue_vals:
                runs = [g for _, g in cell[cell[hue] == h].groupby("run")]
                runs = [g for g in runs if g[metric].notna().any()]
                if not runs:
                    continue
                color = colors[h]
                if aggregate and len(runs) > 1:
                    agg = _aggregate(runs, xcol, metric)
                    if agg is None:
                        continue
                    x, m, sd = agg
                    m = _smooth(m, smooth)
                    ax.plot(x, m, color=color, lw=1.8, label=f"{hue}{h}")
                    if band:
                        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15)
                    last = m[~np.isnan(m)]
                    cand = float(last[-1]) if last.size else np.nan
                else:
                    cand = np.nan
                    for g in runs:
                        x, y = _xy(g, xcol, metric)
                        if len(x) == 0:
                            continue
                        y = _smooth(y, smooth)
                        lbl = f"{hue}{h}" if g is runs[0] else None
                        ax.plot(x, y, color=color, lw=1.4, alpha=0.85, label=lbl)
                        cand = float(y[-1])
                if metric in _LOWER_BETTER and not math.isnan(cand):
                    if best_txt is None or cand < best_txt[1]:
                        best_txt = (h, cand)
            ax.set_ylabel(_LABELS.get(metric, metric))
            ax.grid(alpha=0.25)
            if r == 0 and col is not None:
                ax.set_title(f"{col} = {cv}")
            if r == nrows - 1:
                ax.set_xlabel("epoch" if xcol == "epoch_float" else xcol)
            if best_txt is not None:
                ax.annotate(f"best {hue}{best_txt[0]}: {best_txt[1]:.3g}",
                            xy=(0.98, 0.95), xycoords="axes fraction",
                            ha="right", va="top", fontsize=7,
                            color=colors[best_txt[0]])
    top = 1 - header / fig_h
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle(title, y=1 - 0.22 * header / fig_h, fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1 - 0.62 * header / fig_h),
                   ncol=min(len(labels), 8), frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({nrows}x{ncols} panels, {len(hue_vals)} {hue}s)")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["checkpoints"],
                   help="checkpoint root(s) (or individual run dirs)")
    p.add_argument("-o", "--out", default="training_history.png")
    p.add_argument("--split", choices=["train", "val"], default="train",
                   help="train = per-step loss curves; val = per-epoch eval metrics")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="metric columns to plot (default: a sensible set per split)")
    p.add_argument("--hue", default="variant",
                   help="field for line colour (variant/fold/scheme/peptide_weight/run)")
    p.add_argument("--col", default="peptide_weight",
                   help="field for facet columns ('none' for a single column)")
    p.add_argument("--x", dest="xcol", choices=["epoch", "step", "time"],
                   default="epoch", help="x axis")
    p.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                   help="draw every run instead of mean±std over the grouped dim")
    p.add_argument("--no-band", dest="band", action="store_false",
                   help="hide the ±std band when aggregating")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="moving-average window as a fraction of points (e.g. 0.05)")
    # filters
    p.add_argument("--scheme", nargs="+")
    p.add_argument("--fold", nargs="+", type=int)
    p.add_argument("--variant", nargs="+", type=int)
    p.add_argument("--weight", nargs="+", type=float, help="peptide_weight filter")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = discover([Path(r) for r in args.runs])
    df = _apply_filters(df, args)
    df = df[df["split"] == args.split]
    if df.empty:
        raise SystemExit("no rows after filtering — check --split/--scheme/etc.")

    xcol = {"epoch": "epoch_float", "step": "global_step",
            "time": "wall_time"}.get(args.xcol)

    metrics = args.metrics or (_DEFAULT_TRAIN if args.split == "train"
                               else _DEFAULT_VAL)
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not metrics:
        raise SystemExit("none of the requested metrics have data for this split")

    col = None if (args.col in (None, "none", "None")) else args.col
    if col and df[col].nunique() <= 1:        # don't facet on a constant
        col = None

    title = (f"PMGen-v2 training — {args.split} "
             f"(hue={args.hue}" + (f", col={col}" if col else "") + ")")
    plot(df, metrics, args.hue, col, xcol, args.aggregate, args.band,
         args.smooth, title, Path(args.out))


if __name__ == "__main__":
    main()
