"""
Peptide<->MHC proximity in the data/test PDBs, vs the big H5 training store.

For each data/test structure we take the peptide Cα atoms and, for each, the
distance to the nearest MHC Cα, then summarise per peptide (mean + median nearest).
This is the same quantity ``check_data_quality.py`` computes over the H5 store, so
the two distributions can be overlaid: it tells us whether the small, curated
data/test set places peptides like the bulk training data.

IMPORTANT: the data/test PDBs under ``pdbs/alphafold/`` are **AlphaFold/PMGen
predictions** (they ship ``*_plddt.npy``), *derived from* real PDB entries
(1IM3/3GSO/6UK4) — they are NOT the experimental crystal structures. So this
compares predicted-vs-predicted. For a true experimental reference, fetch the
crystals from the RCSB (separate step).

The peptide is the last chain segment; we split MHC|peptide at the largest
residue-number gap and verify the peptide length matches inputs.tsv.

Run (local):
  python src/post_structure_prediction_processing/check_test_pdb_distance.py \
      --test-dir data/test \
      --store-csv outputs/data_quality/per_example.csv \
      --out-dir outputs/data_quality
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")


def _ca_coords(pdb: Path):
    """Ordered list of (resid, Cα xyz) for the single chain, as written."""
    from Bio.PDB import PDBParser
    s = PDBParser(QUIET=True).get_structure("x", str(pdb))
    out = []
    for model in s:
        for ch in model:
            for res in ch:
                if res.id[0] != " " or "CA" not in res:
                    continue
                out.append((res.id[1], res["CA"].get_coord().astype(float)))
        break
    return out


def _split_mhc_pep(resids, coords, n_pep_expected):
    """Split into (mhc_xyz, pep_xyz). Peptide = the trailing block after the
    largest residue-number jump; fall back to the last n_pep residues."""
    gaps = np.diff(resids)
    cut = int(np.argmax(gaps)) + 1 if len(gaps) else len(resids)
    n_tail = len(resids) - cut
    if n_tail != n_pep_expected:                 # gap split disagreed -> trust length
        cut = len(resids) - n_pep_expected
    return coords[:cut], coords[cut:]


def _nearest(pep, mhc):
    d = np.linalg.norm(pep[:, None, :] - mhc[None, :, :], axis=-1)
    return d.min(axis=1)                         # [n_pep] nearest-MHC per residue


def scan_test(test_dir: Path):
    rows = list(csv.DictReader(open(test_dir / "inputs.tsv"), delimiter="\t"))
    recs, per_res = [], []
    pdb_root = test_dir / "pdbs" / "alphafold"
    for r in rows:
        aid = r["id"]
        n_pep = len(r["peptide"])
        cand = list((pdb_root / aid).glob("*_model_*.pdb"))
        if not cand:
            print(f"  [skip] {aid}: no pdb")
            continue
        ca = _ca_coords(cand[0])
        if len(ca) < n_pep + 5:
            print(f"  [skip] {aid}: too few residues ({len(ca)})")
            continue
        resids = np.array([x[0] for x in ca])
        coords = np.stack([x[1] for x in ca])
        mhc, pep = _split_mhc_pep(resids, coords, n_pep)
        nnd = _nearest(pep, mhc)
        per_res.append(nnd)
        recs.append({
            "id": aid, "base": aid.split("_")[0], "peptide": r["peptide"],
            "n_pep": n_pep, "anchors": r.get("anchors", ""),
            "pep_nndist_mean": float(nnd.mean()),
            "pep_nndist_median": float(np.median(nnd)),
            "pep_nndist_max": float(nnd.max()),
        })
    return recs, (np.concatenate(per_res) if per_res else np.array([]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--store-csv", default="outputs/data_quality/per_example.csv",
                    help="per_example.csv from check_data_quality.py (H5 store)")
    ap.add_argument("--out-dir", default="outputs/data_quality")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs, per_res = scan_test(Path(args.test_dir))
    if not recs:
        print("[test] no structures parsed")
        return
    import pandas as pd
    df = pd.DataFrame(recs)
    df.to_csv(out / "test_pdb_distance.csv", index=False)

    print(f"\n=== data/test peptide->nearest-MHC ({len(df)} predicted structures) ===")
    print(f"  per-peptide MEAN  nearest-MHC: median {df.pep_nndist_mean.median():.2f} "
          f"A  (range {df.pep_nndist_mean.min():.2f}-{df.pep_nndist_mean.max():.2f})")
    print(f"  per-peptide MEDIAN nearest-MHC: median {df.pep_nndist_median.median():.2f} A")
    print(f"  per-peptide MAX   nearest-MHC: median {df.pep_nndist_max.median():.2f} A")
    print("\n  per structure:")
    for _, r in df.iterrows():
        print(f"    {r['id']:<12} {r['peptide']:<14} mean {r['pep_nndist_mean']:.2f} "
              f"median {r['pep_nndist_median']:.2f} max {r['pep_nndist_max']:.2f} A")

    # compare with H5 store
    store = None
    sp = Path(args.store_csv)
    if sp.exists():
        store = pd.read_csv(sp, usecols=["pep_nndist_mean", "pep_nndist_max"])
        print(f"\n=== H5 store ({len(store):,} structures) for reference ===")
        print(f"  per-peptide MEAN nearest-MHC: median {store.pep_nndist_mean.median():.2f} "
              f"A  (p5 {store.pep_nndist_mean.quantile(.05):.2f}, "
              f"p95 {store.pep_nndist_mean.quantile(.95):.2f})")
        print(f"  data/test vs store (mean nearest-MHC): "
              f"{df.pep_nndist_mean.median():.2f} vs {store.pep_nndist_mean.median():.2f} A")
    else:
        print(f"\n[warn] store csv {sp} not found — skipping comparison")

    # plot: overlay distributions + per-structure bars
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    if store is not None:
        ax[0].hist(store.pep_nndist_mean, bins=60, range=(3, 15), density=True,
                   color="#4C72B0", alpha=0.55, label=f"H5 store (n={len(store):,})")
    ax[0].hist(df.pep_nndist_mean, bins=15, range=(3, 15), density=True,
               color="#DD8452", alpha=0.8, label=f"data/test (n={len(df)})")
    if store is not None:
        ax[0].axvline(store.pep_nndist_mean.median(), color="#4C72B0", ls="--", lw=1)
    ax[0].axvline(df.pep_nndist_mean.median(), color="#DD8452", ls="--", lw=1)
    ax[0].set(title="Peptide mean nearest-MHC Cα distance",
              xlabel="distance (Å)", ylabel="density")
    ax[0].legend(fontsize=8)

    order = df.sort_values("pep_nndist_mean")
    ax[1].barh(range(len(order)), order["pep_nndist_mean"], color="#DD8452")
    ax[1].errorbar(order["pep_nndist_mean"], range(len(order)),
                   xerr=[order["pep_nndist_mean"] - order["pep_nndist_median"] * 0,
                         order["pep_nndist_max"] - order["pep_nndist_mean"]],
                   fmt="none", ecolor="#555", elinewidth=0.8, capsize=2)
    ax[1].set_yticks(range(len(order)))
    ax[1].set_yticklabels(order["id"], fontsize=7)
    ax[1].set(title="data/test per structure (bar=mean, whisker→max)",
              xlabel="peptide nearest-MHC distance (Å)")
    ax[1].grid(axis="x", alpha=0.25)
    fig.suptitle("data/test (AF predictions) peptide↔MHC proximity vs H5 store",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "test_pdb_distance.png", dpi=140)
    plt.close(fig)
    print(f"\n[plot] wrote {out/'test_pdb_distance.png'}")
    print(f"[csv]  wrote {out/'test_pdb_distance.csv'}")


if __name__ == "__main__":
    main()
