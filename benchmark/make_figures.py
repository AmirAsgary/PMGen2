"""Box plots comparing model_multimer_1 against the published pRMSD baselines.

Two panels in one figure -- peptide Calpha RMSD and peptide all-atom RMSD -- over
the same 176 complexes, every method scored after superposing on the MHC only.

Design notes: methods share ONE row order (sorted by median Calpha RMSD) so a
method that changes rank between the two panels is visible; colour separates
"this work" from the published baselines and carries no other meaning, since the
row label already gives identity; the raw 176 points are drawn behind each box
because a box alone hides how heavy the tail is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

INK        = "#0b0b0b"
INK_2      = "#52514e"
MUTED      = "#8a8a84"
SURFACE    = "#ffffff"
OURS       = "#2a78d6"     # categorical slot 1
OURS_FILL  = "#d7e6f8"
BASE       = "#9aa0a6"
BASE_FILL  = "#e8e9ea"

BASELINES = ["Pandora", "AF-Multimer", "PMGen+TE", "Tfold",
             "PMGen+plddt", "MHC-Fine", "PMGen"]


def load(ours_csvs: dict) -> tuple:
    """ours_csvs: {label: path}.  Every method is scored on the SAME complexes."""
    bm = pd.read_csv(HERE / "pRMSD_benchmark.tsv", sep="\t")
    long = {m: (bm[f"prmsd_{m}"].to_numpy(), bm[f"prmsd_allatom_{m}"].to_numpy())
            for m in BASELINES}
    for label, path in ours_csvs.items():
        o = pd.read_csv(path)[["PDB", "pep_ca_rmsd", "pep_allatom_rmsd"]]
        d = bm[["PDB"]].merge(o, on="PDB", how="inner", validate="one_to_one")
        if len(d) != len(bm):
            raise SystemExit(f"only {len(d)}/{len(bm)} complexes present in {path}")
        long[label] = (d.pep_ca_rmsd.to_numpy(), d.pep_allatom_rmsd.to_numpy())
    return bm, long


def panel(ax, order, data, xlabel, xmax, mine_set):
    n = len(order)
    ys = np.arange(n)[::-1]
    rng = np.random.default_rng(0)
    n_clip = 0
    for y, m in zip(ys, order):
        v = data[m]
        mine = m in mine_set
        edge, fill = (OURS, OURS_FILL) if mine else (BASE, BASE_FILL)
        jit = y + rng.uniform(-0.17, 0.17, size=len(v))
        vis = v <= xmax
        n_clip += int((~vis).sum())
        ax.scatter(v[vis], jit[vis], s=5, color=edge, alpha=0.22,
                   linewidths=0, zorder=1)
        if (~vis).any():                     # park out-of-range points on the edge
            ax.scatter(np.full((~vis).sum(), xmax * 0.995), jit[~vis], s=9,
                       marker="4", color=edge, alpha=0.8, linewidths=0.9, zorder=2)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.52,
                        showfliers=False, patch_artist=True, zorder=3,
                        medianprops=dict(color=edge, linewidth=2.4),
                        boxprops=dict(facecolor=fill, edgecolor=edge,
                                      linewidth=1.6, alpha=0.92),
                        whiskerprops=dict(color=edge, linewidth=1.4),
                        capprops=dict(color=edge, linewidth=1.4))
        med = float(np.median(v))
        ax.text(1.115, y, f"{med:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9,
                color=INK if mine else INK_2,
                fontweight="bold" if mine else "normal", clip_on=False)
    ax.text(1.115, n - 0.30, "median", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, color=MUTED, clip_on=False)

    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=9.5)
    for t, m in zip(ax.get_yticklabels(), order):
        if m in mine_set:
            t.set_color(INK); t.set_fontweight("bold")
        else:
            t.set_color(INK_2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlabel(xlabel, fontsize=10, color=INK_2)
    ax.xaxis.grid(True, color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d6d2")
    ax.tick_params(axis="x", colors=INK_2, labelsize=9, length=3)
    ax.tick_params(axis="y", length=0)
    return n_clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_mm1_crystal.csv"))
    ap.add_argument("--results-template", default=str(HERE / "results_mm1_template.csv"))
    ap.add_argument("--label", default="PMGen2-distilled (this work)")
    ap.add_argument("--label-template",
                    default="PMGen2-distilled, foreign MHC groove")
    ap.add_argument("--out", default=str(HERE / "benchmark_boxplots"))
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    ours = {args.label: Path(args.results)}
    mine_set = {args.label}
    if args.results_template and Path(args.results_template).exists():
        ours[args.label_template] = Path(args.results_template)
        mine_set.add(args.label_template)
    df, long = load(ours)
    ca = {m: v[0] for m, v in long.items()}
    aa = {m: v[1] for m, v in long.items()}
    order = sorted(ca, key=lambda m: np.median(ca[m]))          # best at top

    # ---- tidy CSV of every method x every complex, plus the summary table ----
    tidy = []
    for m in order:
        for pdb, c, a in zip(df.PDB, ca[m], aa[m]):
            tidy.append(dict(PDB=pdb, method=m, pep_ca_rmsd=c, pep_allatom_rmsd=a))
    tidy = pd.DataFrame(tidy)
    tidy.to_csv(HERE / "benchmark_all_methods_long.csv", index=False)

    summ = []
    for m in order:
        c, a = ca[m], aa[m]
        summ.append(dict(
            method=m, n=len(c),
            ca_mean=c.mean(), ca_median=np.median(c),
            ca_q1=np.percentile(c, 25), ca_q3=np.percentile(c, 75),
            ca_frac_under_1A=(c < 1).mean(), ca_frac_under_2A=(c < 2).mean(),
            allatom_mean=a.mean(), allatom_median=np.median(a),
            allatom_q1=np.percentile(a, 25), allatom_q3=np.percentile(a, 75),
            allatom_frac_under_2A=(a < 2).mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(HERE / "benchmark_summary.csv", index=False)

    # ---- figure ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), constrained_layout=False)
    c1 = panel(axes[0], order, ca, "peptide Cα RMSD (Å)", 6.0, mine_set)
    c2 = panel(axes[1], order, aa, "peptide all-atom RMSD (Å)", 8.0, mine_set)
    axes[1].set_yticklabels([])

    fig.suptitle("Peptide accuracy on the pRMSD benchmark "
                 f"({len(df)} pMHC-I complexes, MHC-superposed)",
                 fontsize=13, color=INK, x=0.5, y=0.985, ha="center")
    sub = args.subtitle or ("box = quartiles, line = median (labelled); "
                            "each dot is one complex; methods ordered by median Cα")
    fig.text(0.5, 0.925, sub, fontsize=9, color=MUTED, ha="center")
    fig.text(0.5, 0.022,
             f"axes clipped at 6 Å / 8 Å; {c1} and {c2} points lie beyond and are "
             f"drawn as ticks on the right edge.  All methods scored on the identical "
             f"{len(df)} complexes.",
             fontsize=7.8, color=MUTED, ha="center")
    fig.subplots_adjust(left=0.22, right=0.925, top=0.855, bottom=0.185, wspace=0.16)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor=SURFACE)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nwrote {args.out}.png / .pdf")
    print(f"wrote {HERE/'benchmark_summary.csv'}")
    print(f"wrote {HERE/'benchmark_all_methods_long.csv'}")


if __name__ == "__main__":
    main()
