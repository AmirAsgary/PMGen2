"""
Post-structure-prediction cleaning.

Exclude every example whose PMGen structure (pdb / plddt / pae) does not exist
from the training pool AND the cross-validation / test splits.

Source of truth = the HDF5 store ``index.csv`` (``preprocess.py`` only records an
id whose pdb+plddt+pae existed and parsed). So:
  - an *anchor* id is VALID iff it is in the index;
  - a *base* id (peptide-MHC pair) is usable iff >= 1 of its anchor combos is
    valid; a base id with 0 valid anchors is dropped from every split.

This writes ``data/processed/valid_base_ids.csv``, which ``read_split_ids`` (and
thus all of build_dataset / build_h5_dataset / training) intersects with — so the
exclusion takes effect *everywhere*. It also writes the excluded lists + a
per-(scheme, fold, split) survival report.

Run:
  python src/post_structure_prediction_processing/script.py \
      --h5-dir data/processed/h5_store
  # optional: --anchors-tsv outputs/pmgen_input/full_dataset/Multiple_Anchors_input_reduced.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def base_id(anchor_id: str) -> str:
    """'BOLA100901_ELGNITGN_8_3' -> 'BOLA100901_ELGNITGN_8'."""
    return anchor_id.rsplit("_", 1)[0]


def _row_idx(path: Path) -> list[int]:
    return pd.read_csv(path)["row_idx"].tolist()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", default="data/processed/h5_store",
                    help="store dir holding index.csv (the valid structures)")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--base-ids", default=None,
                    help="ordered base-id list (default <processed-dir>/base_ids_ordered.csv)")
    ap.add_argument("--anchors-tsv", default=None,
                    help="reduced anchors TSV, to also list excluded anchor ids")
    ap.add_argument("--out-dir", default=None,
                    help="reports dir (default <processed-dir>/post_structure_prediction)")
    ap.add_argument("--schemes", nargs="+", default=["two_axis", "hla_only"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args(argv)

    proc = Path(args.processed_dir)
    base_ids_csv = Path(args.base_ids) if args.base_ids else proc / "base_ids_ordered.csv"
    out_dir = Path(args.out_dir) if args.out_dir else proc / "post_structure_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- valid structures = the H5 index ----
    idx = pd.read_csv(Path(args.h5_dir) / "index.csv", dtype=str)
    valid_anchor = set(idx["id"])
    if "base_id" in idx.columns:
        valid_base = set(idx["base_id"])
        anchors_per_base = idx.groupby("base_id")["id"].size()
    else:
        idx["base_id"] = idx["id"].map(base_id)
        valid_base = set(idx["base_id"])
        anchors_per_base = idx.groupby("base_id")["id"].size()
    print(f"[clean] valid: {len(valid_anchor):,} anchor structures across "
          f"{len(valid_base):,} base ids")

    # ---- pool base ids (ordered) -> which have no structure at all ----
    pool = pd.read_csv(base_ids_csv, dtype=str)["id"].tolist()
    pool_set = set(pool)
    valid_pool = [b for b in pool if b in valid_base]           # keep pool order
    dropped_base = sorted(pool_set - valid_base)
    print(f"[clean] pool base ids: {len(pool_set):,} | with >=1 structure: "
          f"{len(valid_pool):,} | fully dropped: {len(dropped_base):,}")

    # ---- authoritative valid list (read_split_ids honors it) + excluded lists ----
    pd.DataFrame({"id": valid_pool}).to_csv(proc / "valid_base_ids.csv", index=False)
    pd.DataFrame({"id": dropped_base}).to_csv(out_dir / "excluded_base_ids.csv",
                                              index=False)
    print(f"[clean] wrote {proc / 'valid_base_ids.csv'} ({len(valid_pool):,}) and "
          f"{out_dir / 'excluded_base_ids.csv'} ({len(dropped_base):,})")

    if args.anchors_tsv:
        all_anchor = set(pd.read_csv(args.anchors_tsv, sep="\t",
                                     usecols=["id"], dtype=str)["id"])
        excluded_anchor = sorted(all_anchor - valid_anchor)
        pd.DataFrame({"id": excluded_anchor}).to_csv(
            out_dir / "excluded_anchor_ids.csv", index=False)
        print(f"[clean] anchors: {len(all_anchor):,} total | valid "
              f"{len(valid_anchor):,} | excluded {len(excluded_anchor):,} -> "
              f"{out_dir / 'excluded_anchor_ids.csv'}")

    # ---- per-(scheme, fold, split) survival report ----
    def split_stats(bset: set) -> tuple[int, int, int]:
        kept = bset & valid_base
        n_anchor = int(anchors_per_base.reindex(list(kept)).fillna(0).sum())
        return len(bset), len(kept), n_anchor

    rows = []
    for scheme in args.schemes:
        sdir = proc / scheme
        if not (sdir / "test" / "test.csv").exists():
            continue
        test_base = {pool[i] for i in _row_idx(sdir / "test" / "test.csv")}
        for fold in args.folds:
            vp = sdir / "cv" / f"fold_{fold}" / "val.csv"
            if not vp.exists():
                continue
            val_base = {pool[i] for i in _row_idx(vp)}
            train_base = pool_set - test_base - val_base
            for name, bset in (("train", train_base), ("val", val_base),
                               ("test", test_base)):
                nb, nk, na = split_stats(bset)
                rows.append({"scheme": scheme, "fold": fold, "split": name,
                             "base_ids": nb, "base_ids_with_structure": nk,
                             "dropped_base_ids": nb - nk, "anchor_examples": na})
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "report.csv", index=False)
    print("\n[clean] per-split survival:")
    print(report.to_string(index=False))
    print(f"\n[clean] report -> {out_dir / 'report.csv'}")
    print("[clean] done. read_split_ids now intersects with valid_base_ids.csv, so "
          "failed structures are excluded from train/val/test everywhere.")


if __name__ == "__main__":
    main()
