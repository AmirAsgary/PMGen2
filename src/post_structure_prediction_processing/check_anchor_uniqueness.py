"""
Confirm there is no anchor-register CONFLATION in the store.

model_1 conveys the peptide's alignment register only through the anchor flag, so
the question is: does any (peptide, MHC) **base id** have two anchor-ids that share
the *same anchor positions*? If yes, those are different teacher structures with an
identical model input -> the model would average them (blurry peptide). If every
anchor-id of a base has a unique anchor set, there is zero ambiguity.

This reads the per-id ``anchors`` attribute straight from the HDF5 store (no
re-preprocess) and reports duplicates per base id.

Run:
  python src/post_structure_prediction_processing/check_anchor_uniqueness.py \
      --h5-dir data/processed/h5_store_sc            # or h5_store
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import pandas as pd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-dir", default="data/processed/h5_store")
    ap.add_argument("--out", default=None, help="optional CSV of the conflated bases")
    args = ap.parse_args(argv)

    h5_dir = Path(args.h5_dir)
    idx = pd.read_csv(h5_dir / "index.csv", dtype=str)
    by_shard = defaultdict(list)
    for _, r in idx.iterrows():
        by_shard[r["shard"]].append((r["id"], r["base_id"]))

    # base_id -> {anchor_string: [ids]}
    base_anchor = defaultdict(lambda: defaultdict(list))
    n_ids = 0
    for shard, items in by_shard.items():
        with h5py.File(h5_dir / shard, "r") as h5:
            for aid, base in items:
                if aid not in h5:
                    continue
                anchors = str(h5[aid].attrs.get("anchors", ""))
                base_anchor[base][anchors].append(aid)
                n_ids += 1

    conflated = []           # (base, anchors, [ids]) where >1 id shares an anchor set
    for base, groups in base_anchor.items():
        for anchors, ids in groups.items():
            if len(ids) > 1:
                conflated.append((base, anchors, ids))

    n_bases = len(base_anchor)
    n_conf_bases = len({c[0] for c in conflated})
    n_conf_pairs = sum(len(ids) - 1 for _, _, ids in conflated)
    print(f"[anchors] scanned {n_ids:,} anchor-ids across {n_bases:,} base ids")
    print(f"[anchors] base ids with >=2 anchor-ids sharing identical anchors: "
          f"{n_conf_bases:,}")
    print(f"[anchors] conflated id-pairs (same input, different structure): "
          f"{n_conf_pairs:,}")
    if conflated:
        print("[anchors] examples (base | anchors | ids):")
        for base, anchors, ids in conflated[:10]:
            print(f"    {base} | '{anchors}' | {ids}")
        print("[anchors] => these registers ARE ambiguous to model_1; the gap-aware "
              "relpos (--anchor-relpos) would help here.")
    else:
        print("[anchors] => CLEAN: every anchor-id has a unique anchor set, so the "
              "anchor flag fully disambiguates the registers. No conflation.")
    if args.out and conflated:
        pd.DataFrame([(b, a, ";".join(i)) for b, a, i in conflated],
                     columns=["base_id", "anchors", "ids"]).to_csv(args.out, index=False)
        print(f"[anchors] wrote {args.out}")


if __name__ == "__main__":
    main()
