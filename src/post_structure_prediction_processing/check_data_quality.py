"""
Data-quality audit of the teacher (PMGen/AF2) structures in the H5 store.

Answers three questions, to tell whether poor model performance is a *data*
problem rather than a *model* problem:

  1. Peptide pLDDT distribution  -- how confident is the teacher about the part
     we actually care about (the peptide)?  Low peptide pLDDT => the teacher
     itself is unsure where the peptide is, so there is no clean signal to learn.

  2. Peptide<->MHC proximity     -- for every peptide Calpha, the distance to its
     nearest MHC Calpha.  A peptide bound in the groove has every residue within
     ~4-9 A of the MHC walls; a residue that dangles >~12 A is outside the
     groove (teacher placed it badly / it is unbound).  We report, per peptide,
     the mean and the MAX nearest-MHC distance, and call a peptide "outside" if
     its max exceeds --outside-thresh.

  3. (1) and (2) stratified by HLA cluster (index.csv `hla_cluster_id`), so we
     can see whether specific alleles carry the bad data.

Everything is read straight from the store (Calpha + pLDDT + segment id); no model
and no re-preprocess.  Peptide = highest segment id in each example (same rule the
loaders use).

Run (on the cluster, in the repo root, pmgen2 env):
  python src/post_structure_prediction_processing/check_data_quality.py \
      --h5-dir data/processed/h5_store_sc \
      --out-dir outputs/data_quality
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _peptide_mask(segment_id: np.ndarray) -> np.ndarray:
    """Peptide = highest segment id (matches pyg_data / utils convention)."""
    return segment_id == segment_id.max()


def _nearest_mhc_dist(ca: np.ndarray, pep: np.ndarray) -> np.ndarray:
    """For each peptide Calpha, the min Euclidean distance to any MHC Calpha. [n_pep]."""
    p = ca[pep]                      # [n_pep, 3]
    m = ca[~pep]                     # [n_mhc, 3]
    if len(m) == 0 or len(p) == 0:
        return np.full(len(p), np.nan, dtype=np.float32)
    d = np.linalg.norm(p[:, None, :] - m[None, :, :], axis=-1)   # [n_pep, n_mhc]
    return d.min(axis=1)


def scan(h5_dir: Path, outside_thresh: float):
    idx = pd.read_csv(h5_dir / "index.csv", dtype=str)
    by_shard = defaultdict(list)
    for _, r in idx.iterrows():
        by_shard[r["shard"]].append(
            (r["id"], r.get("hla_cluster_id", ""), r.get("peptide_cluster_id", "")))

    rows = []
    res_plddt = []                   # per-residue peptide pLDDT (for the histogram)
    res_nndist = []                  # per-residue nearest-MHC distance
    n_missing = 0
    for shard, items in by_shard.items():
        path = h5_dir / shard
        if not path.exists():
            n_missing += len(items)
            continue
        with h5py.File(path, "r") as h5:
            for aid, hla, pepc in items:
                if aid not in h5:
                    n_missing += 1
                    continue
                g = h5[aid]
                seg = g["segment_id"][()].astype(np.int64)
                ca = g["teacher_ca"][()].astype(np.float32)
                plddt = g["teacher_plddt"][()].astype(np.float32)
                pep = _peptide_mask(seg)
                if pep.sum() == 0:
                    continue
                pep_plddt = plddt[pep]
                nnd = _nearest_mhc_dist(ca, pep)
                res_plddt.append(pep_plddt)
                res_nndist.append(nnd)
                rows.append({
                    "id": aid,
                    "hla_cluster_id": hla,
                    "peptide_cluster_id": pepc,
                    "n_pep": int(pep.sum()),
                    "n_mhc": int((~pep).sum()),
                    "pep_plddt_mean": float(np.nanmean(pep_plddt)),
                    "pep_plddt_min": float(np.nanmin(pep_plddt)),
                    "pep_nndist_mean": float(np.nanmean(nnd)),
                    "pep_nndist_max": float(np.nanmax(nnd)),
                    "outside": bool(np.nanmax(nnd) > outside_thresh),
                })
    df = pd.DataFrame(rows)
    res_plddt = np.concatenate(res_plddt) if res_plddt else np.array([])
    res_nndist = np.concatenate(res_nndist) if res_nndist else np.array([])
    return df, res_plddt, res_nndist, n_missing


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def report(df: pd.DataFrame, outside_thresh: float):
    n = len(df)
    print(f"\n=== DATA QUALITY: {n:,} examples ===")
    print(f"\n[1] PEPTIDE pLDDT (teacher confidence on the peptide)")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:>2} per-example mean = {np.percentile(df['pep_plddt_mean'], q):6.1f}")
    print(f"    examples with mean peptide pLDDT < 50 : "
          f"{(df['pep_plddt_mean'] < 50).mean()*100:5.1f}%  (low confidence)")
    print(f"    examples with mean peptide pLDDT < 70 : "
          f"{(df['pep_plddt_mean'] < 70).mean()*100:5.1f}%")
    print(f"    examples with mean peptide pLDDT >=90 : "
          f"{(df['pep_plddt_mean'] >= 90).mean()*100:5.1f}%  (high confidence)")

    print(f"\n[2] PEPTIDE<->MHC PROXIMITY (groove placement)")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:>2} per-residue nearest-MHC = "
              f"{np.percentile(df['pep_nndist_mean'], q):6.2f} A  (per-example mean)")
    print(f"    median of per-peptide MAX nearest-MHC = "
          f"{df['pep_nndist_max'].median():.2f} A")
    print(f"    peptides OUTSIDE groove (max nearest-MHC > {outside_thresh:g} A): "
          f"{df['outside'].mean()*100:5.1f}%  ({int(df['outside'].sum()):,}/{n:,})")


def report_strata(df: pd.DataFrame, by: str, outside_thresh: float, top: int):
    g = df.groupby(by)
    summary = g.agg(
        n=("id", "size"),
        plddt_mean=("pep_plddt_mean", "mean"),
        plddt_med=("pep_plddt_mean", "median"),
        nndist_med=("pep_nndist_mean", "median"),
        nndist_max_med=("pep_nndist_max", "median"),
        frac_outside=("outside", "mean"),
        frac_lowconf=("pep_plddt_mean", lambda s: (s < 50).mean()),
    ).reset_index()
    summary = summary.sort_values("n", ascending=False)
    print(f"\n[3] STRATIFIED BY {by}  ({summary.shape[0]:,} groups; "
          f"showing {min(top, len(summary))} largest)")
    print(f"    {'group':<22} {'n':>5} {'plddt':>6} {'nndist':>7} "
          f"{'max_nnd':>7} {'%out':>6} {'%low':>6}")
    for _, r in summary.head(top).iterrows():
        print(f"    {str(r[by])[:22]:<22} {int(r['n']):>5} "
              f"{r['plddt_med']:>6.1f} {r['nndist_med']:>7.2f} "
              f"{r['nndist_max_med']:>7.2f} {r['frac_outside']*100:>5.1f}% "
              f"{r['frac_lowconf']*100:>5.1f}%")
    print(f"    WORST 5 clusters by median peptide pLDDT (>=10 examples):")
    big = summary[summary["n"] >= 10].sort_values("plddt_med")
    for _, r in big.head(5).iterrows():
        print(f"      {str(r[by])[:22]:<22} n={int(r['n']):>4} "
              f"plddt={r['plddt_med']:.1f} nndist={r['nndist_med']:.2f} "
              f"%out={r['frac_outside']*100:.0f}")
    return summary


def make_plots(df, res_plddt, res_nndist, strata, by, out_dir, outside_thresh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].hist(res_plddt, bins=50, range=(0, 100), color="#4C72B0")
    ax[0, 0].axvline(50, color="r", ls="--", lw=1, label="50 (low)")
    ax[0, 0].axvline(70, color="orange", ls="--", lw=1, label="70")
    ax[0, 0].set(title="[1] Peptide pLDDT (per residue)", xlabel="pLDDT", ylabel="residues")
    ax[0, 0].legend(fontsize=8)

    nn = res_nndist[np.isfinite(res_nndist)]
    ax[0, 1].hist(nn, bins=60, range=(0, 30), color="#55A868")
    ax[0, 1].axvline(outside_thresh, color="r", ls="--", lw=1,
                     label=f"{outside_thresh:g} A (outside)")
    ax[0, 1].set(title="[2] Peptide residue -> nearest MHC Calpha",
                 xlabel="distance (A)", ylabel="residues")
    ax[0, 1].legend(fontsize=8)

    # per-cluster scatter: pLDDT vs proximity, size ~ n (only the larger groups)
    big = strata[strata["n"] >= 5]
    sc = ax[1, 0].scatter(big["plddt_med"], big["nndist_med"],
                          s=np.clip(big["n"], 5, 300), alpha=0.5,
                          c=big["frac_outside"], cmap="Reds", vmin=0, vmax=1)
    ax[1, 0].set(title=f"[3] per {by} (size~n, color=%outside)",
                 xlabel="median peptide pLDDT", ylabel="median nearest-MHC (A)")
    fig.colorbar(sc, ax=ax[1, 0], label="frac outside")

    # distribution of per-cluster median peptide pLDDT
    ax[1, 1].hist(strata["plddt_med"], bins=40, color="#C44E52")
    ax[1, 1].set(title=f"[3] median peptide pLDDT across {by}s",
                 xlabel="median peptide pLDDT", ylabel=f"# {by}s")
    fig.tight_layout()
    p = out_dir / "data_quality.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n[plot] wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    ap.add_argument("--out-dir", default="outputs/data_quality")
    ap.add_argument("--stratify-by", default="hla_cluster_id",
                    choices=["hla_cluster_id", "peptide_cluster_id"])
    ap.add_argument("--outside-thresh", type=float, default=12.0,
                    help="peptide is 'outside groove' if its MAX residue "
                         "nearest-MHC distance exceeds this (A)")
    ap.add_argument("--top", type=int, default=25,
                    help="how many largest strata to print")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, res_plddt, res_nndist, n_missing = scan(h5_dir, args.outside_thresh)
    if df.empty:
        print(f"[error] no examples read from {h5_dir} (missing={n_missing})")
        return
    if n_missing:
        print(f"[warn] {n_missing} ids missing from shards (skipped)")

    report(df, args.outside_thresh)
    strata = report_strata(df, args.stratify_by, args.outside_thresh, args.top)

    df.to_csv(out_dir / "per_example.csv", index=False)
    strata.to_csv(out_dir / f"by_{args.stratify_by}.csv", index=False)
    print(f"\n[csv] per-example  -> {out_dir/'per_example.csv'}")
    print(f"[csv] per-stratum  -> {out_dir/('by_'+args.stratify_by+'.csv')}")

    if not args.no_plots:
        try:
            make_plots(df, res_plddt, res_nndist, strata,
                       args.stratify_by, out_dir, args.outside_thresh)
        except Exception as e:
            print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main()
