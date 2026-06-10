"""
STEP 3 sanity checks for the PMGen-v2 PDB parser.

Runs ``parse_example`` over every entry in data/test/inputs.tsv and asserts the
structural invariants (lengths, the single MHC->peptide numbering gap at the
segment boundary, anchors confined to the peptide, monotone segment_id with
boundaries matching n_mhc/n_pep). Prints residue_index and the anchor vector for
each file plus a one-line summary.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/pdb/check_parse.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

import parse  # same directory
from parse import GAP_THRESHOLD, REPO_ROOT

TEST_DIR = REPO_ROOT / "data" / "test"


def _ranges(values: list[int]) -> str:
    """Compact a sorted int list into 'a..b, c..d' contiguous runs."""
    if not values:
        return "[]"
    runs, start, prev = [], values[0], values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append(f"{start}..{prev}" if start != prev else f"{start}")
        start = prev = v
    runs.append(f"{start}..{prev}" if start != prev else f"{start}")
    return ", ".join(runs)


def check_one(fid: str, row: dict[str, str]) -> dict[str, int]:
    pdb = TEST_DIR / "pdbs" / "alphafold" / fid / f"{fid}_model_1_model_2_ptm.pdb"
    ex = parse.parse_example(pdb, row["peptide"], row["mhc_seq"],
                             row["anchors"], row["mhc_type"])
    n_mhc, n_pep = ex["n_mhc"], ex["n_pep"]
    n = int(ex["aatype"].shape[0])
    ri = ex["residue_index"]
    seg = ex["segment_id"]
    anchor_idx = (ex["anchor"] == 1).nonzero().squeeze(-1).tolist()

    # --- length agreement ---
    assert n_mhc + n_pep == n, f"{fid}: n_mhc+n_pep {n_mhc+n_pep} != N {n}"
    for k in ("residue_index", "seq_mask", "anchor", "teacher_ca", "segment_id"):
        assert ex[k].shape[0] == n, f"{fid}: {k} length {ex[k].shape[0]} != N {n}"

    # --- exactly one large jump, located at the boundary (index n_mhc) ---
    jumps = (ri[1:] - ri[:-1] >= GAP_THRESHOLD).nonzero().squeeze(-1).tolist()
    assert jumps == [n_mhc - 1], (
        f"{fid}: large jumps at {jumps}, expected only [{n_mhc - 1}] "
        f"(boundary before peptide index {n_mhc})"
    )

    # --- anchors only in the peptide region ---
    assert all(i >= n_mhc for i in anchor_idx), \
        f"{fid}: anchor indices {anchor_idx} include MHC region (< {n_mhc})"

    # --- segment_id monotone non-decreasing, boundaries match n_mhc/n_pep ---
    assert torch.all(seg[1:] >= seg[:-1]), f"{fid}: segment_id not non-decreasing"
    assert seg.unique().tolist() == [0, 1], f"{fid}: class-I segment_id != {{0,1}}"
    assert int((seg == 0).sum()) == n_mhc and int((seg == 1).sum()) == n_pep, \
        f"{fid}: segment counts {(int((seg==0).sum()), int((seg==1).sum()))} " \
        f"!= (n_mhc {n_mhc}, n_pep {n_pep})"

    print(f"--- {fid} (mhc_type={row['mhc_type']}, anchors={row['anchors']}) ---")
    print(f"  residue_index: {_ranges(ri.tolist())}   (boundary "
          f"{int(ri[n_mhc-1])}->{int(ri[n_mhc])}, jump {int(ri[n_mhc]-ri[n_mhc-1])})")
    print(f"  anchor==1 at indices {anchor_idx}  -> peptide positions "
          f"{[i - n_mhc + 1 for i in anchor_idx]} (1-indexed)")
    return {"n_mhc": n_mhc, "n_pep": n_pep, "n_anchors": len(anchor_idx),
            "mhc_type": int(row["mhc_type"])}


def main() -> None:
    with open(TEST_DIR / "inputs.tsv") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    summaries: list[tuple[str, dict[str, int]]] = []
    for fid in sorted(rows):
        summaries.append((fid, check_one(fid, rows[fid])))

    print("\n===== SUMMARY (all assertions passed) =====")
    print(f"{'id':10} {'mhc_type':9} {'n_mhc':6} {'n_pep':6} {'n_anchors':9}")
    for fid, s in summaries:
        print(f"{fid:10} {s['mhc_type']:<9} {s['n_mhc']:<6} {s['n_pep']:<6} "
              f"{s['n_anchors']:<9}")
    print(f"\n{len(summaries)} files OK.")


if __name__ == "__main__":
    main()
