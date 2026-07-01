"""
data_exploration.py — peptide pLDDT + MHC-coordination BURIAL SCORE over the teacher
H5 store. One pass, no model. Writes plots + a per-structure CSV.

Burial score (MHC coordination number). For peptide Cα p_1..p_L and MHC Cα m_1..m_M,
per peptide residue i:
    coord_i  = |{ j : ||p_i - m_j|| < R }|          (# MHC Cα within R Å)
    buried_i = sigmoid((coord_i - c0) / s_c)        in [0,1]
with R=10 Å, c0=6, s_c=2. Unlike a nearest-neighbour distance, this measures the local
DENSITY of surrounding MHC — deeply grooved residues score ~1, weakly-contacting ones
~0 even if their nearest MHC atom is close. Per-structure burial_score = mean over the
peptide residues.

Outputs (into --out-dir):
  data_exploration.png              -- pLDDT + burial distributions + joint scatter
  per_structure.csv                 -- id, anchor, mean_peptide_plddt, burial_score

Run (cluster, pmgen2 env):
  python src/visualization/data_exploration.py --h5-dir data/processed/h5_store \
      --out-dir outputs/data_exploration
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _anchor_str(g) -> str:
    """Anchor string from the group attr; fall back to the per-residue anchor array
    (1-based positions within the peptide)."""
    a = str(g.attrs.get("anchors", "") or "")
    if a:
        return a
    if "anchor" in g and "segment_id" in g:
        seg = g["segment_id"][()].astype(np.int64)
        anc = g["anchor"][()].astype(np.int64)
        pep = seg == seg.max()
        pep_idx = np.where(pep)[0]
        pos = [k + 1 for k, i in enumerate(pep_idx) if anc[i] > 0]   # 1-based in peptide
        return ";".join(map(str, pos))
    return ""


def scan(h5_dir: Path, R: float, c0: float, s_c: float, max_structs: int = 0):
    idx = pd.read_csv(h5_dir / "index.csv", dtype=str)
    by_shard = defaultdict(list)
    for _, r in idx.iterrows():
        by_shard[r["shard"]].append(r["id"])

    rows = []
    res_plddt, res_buried = [], []          # per-residue, for the distributions
    n_missing = 0
    for shard, ids in by_shard.items():
        path = h5_dir / shard
        if not path.exists():
            n_missing += len(ids)
            continue
        with h5py.File(path, "r") as h5:
            for aid in ids:
                if aid not in h5:
                    n_missing += 1
                    continue
                g = h5[aid]
                seg = g["segment_id"][()].astype(np.int64)
                ca = g["teacher_ca"][()].astype(np.float32)
                plddt = g["teacher_plddt"][()].astype(np.float32)
                pep = seg == seg.max()                       # peptide = highest segment
                if pep.sum() == 0 or (~pep).sum() == 0:
                    continue
                pep_ca, mhc_ca = ca[pep], ca[~pep]
                # coordination number: # MHC Cα within R of each peptide Cα
                d = np.linalg.norm(pep_ca[:, None, :] - mhc_ca[None, :, :], axis=-1)
                coord = (d < R).sum(axis=1)                  # [L]
                buried = _sigmoid((coord - c0) / s_c)        # [L] in [0,1]
                pep_plddt = plddt[pep]
                rows.append({
                    "id": aid,
                    "anchor": _anchor_str(g),
                    "mean_peptide_plddt": float(np.nanmean(pep_plddt)),
                    "burial_score": float(np.mean(buried)),
                })
                res_plddt.append(pep_plddt)
                res_buried.append(buried)
                if max_structs and len(rows) >= max_structs:
                    break
        if max_structs and len(rows) >= max_structs:
            break
    df = pd.DataFrame(rows)
    res_plddt = np.concatenate(res_plddt) if res_plddt else np.array([])
    res_buried = np.concatenate(res_buried) if res_buried else np.array([])
    return df, res_plddt, res_buried, n_missing


def report(df: pd.DataFrame):
    n = len(df)
    print(f"\n=== DATA EXPLORATION: {n:,} structures ===")
    print("\n[peptide pLDDT] per-structure mean")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:>2} = {np.percentile(df['mean_peptide_plddt'], q):6.1f}")
    print(f"  %<50 = {(df['mean_peptide_plddt'] < 50).mean()*100:5.1f}   "
          f"%>=70 = {(df['mean_peptide_plddt'] >= 70).mean()*100:5.1f}")
    print("\n[burial score] per-structure mean")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:>2} = {np.percentile(df['burial_score'], q):6.3f}")
    print(f"  %<0.5 (weakly buried / exposed) = {(df['burial_score'] < 0.5).mean()*100:5.1f}")
    print(f"  corr(mean pLDDT, burial_score)  = "
          f"{df['mean_peptide_plddt'].corr(df['burial_score']):+.3f}")


def make_plots(df, res_plddt, res_buried, out_dir: Path, c0, s_c, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # peptide pLDDT
    ax[0, 0].hist(res_plddt, bins=50, range=(0, 100), color="#4C72B0")
    ax[0, 0].axvline(50, color="r", ls="--", lw=1, label="50")
    ax[0, 0].axvline(70, color="orange", ls="--", lw=1, label="70")
    ax[0, 0].set(title="peptide pLDDT (per residue)", xlabel="pLDDT", ylabel="residues")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].hist(df["mean_peptide_plddt"], bins=50, range=(0, 100), color="#4C72B0")
    ax[0, 1].set(title="mean peptide pLDDT (per structure)",
                 xlabel="mean pLDDT", ylabel="structures")

    # burial score
    ax[1, 0].hist(res_buried, bins=40, range=(0, 1), color="#55A868")
    ax[1, 0].axvline(0.5, color="r", ls="--", lw=1)
    ax[1, 0].set(title=f"burial score (per residue)  R={R:g} c0={c0:g} s={s_c:g}",
                 xlabel="buried  (0=exposed, 1=grooved)", ylabel="residues")
    ax[1, 1].scatter(df["mean_peptide_plddt"], df["burial_score"], s=5, alpha=0.2,
                     color="#8172B3")
    ax[1, 1].set(title="joint: pLDDT vs burial (per structure)",
                 xlabel="mean peptide pLDDT", ylabel="burial score")
    ax[1, 1].grid(alpha=0.25)
    fig.suptitle("Teacher data: peptide pLDDT + MHC-coordination burial", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "data_exploration.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n[plot] wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", default="data/processed/h5_store")
    ap.add_argument("--out-dir", default="outputs/data_exploration")
    ap.add_argument("--radius", type=float, default=10.0, help="R: coordination sphere (Å)")
    ap.add_argument("--c0", type=float, default=6.0, help="coordination midpoint")
    ap.add_argument("--s-c", dest="s_c", type=float, default=2.0, help="softness")
    ap.add_argument("--max", type=int, default=0, help="cap #structures (0=all; for testing)")
    args = ap.parse_args(argv)

    h5_dir, out_dir = Path(args.h5_dir), Path(args.out_dir)
    df, res_plddt, res_buried, n_missing = scan(
        h5_dir, args.radius, args.c0, args.s_c, args.max)
    if df.empty:
        print(f"[error] no structures read from {h5_dir} (missing={n_missing})")
        return
    if n_missing:
        print(f"[warn] {n_missing} ids missing from shards (skipped)")

    report(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "per_structure.csv"
    df[["id", "anchor", "mean_peptide_plddt", "burial_score"]].to_csv(csv, index=False)
    print(f"[csv]  wrote {csv}  ({len(df):,} rows)")
    make_plots(df, res_plddt, res_buried, out_dir, args.c0, args.s_c, args.radius)


if __name__ == "__main__":
    main()
