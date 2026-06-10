"""
GATE 2 — data layer verification (PART 2).

Inspects one alphafold_output_path dir (filenames + that PAE is an N×N Å matrix
and pLDDT is per-residue), builds a --dummy Dataset item and a padded batch, and
asserts shapes/masks (including correct padding of the teacher arrays). Also
smoke-checks the scheme/fold split builder against the processed CSVs.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/model/data_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402


def inspect_one_dir() -> None:
    rows = U.dummy_rows()
    r = rows[0]
    af = Path(r["alphafold_output_path"])
    print(f"=== inspect alphafold_output_path: {af.name} ===")
    print("  files:", sorted(p.name for p in af.iterdir()))
    pdb, plddt_npy, pae_npy = U.find_teacher_files(af)
    print(f"  pdb   : {pdb.name}")
    print(f"  plddt : {plddt_npy.name}")
    print(f"  pae   : {pae_npy.name}")
    plddt = np.load(plddt_npy)
    pae = np.load(pae_npy)
    print(f"  plddt: shape {plddt.shape} (per-residue), range "
          f"[{plddt.min():.1f}, {plddt.max():.1f}]  -> 0..100 pLDDT")
    print(f"  pae  : shape {pae.shape} (N×N), range "
          f"[{pae.min():.2f}, {pae.max():.2f}] Å, diag mean "
          f"{np.diagonal(pae).mean():.2f} Å")
    assert plddt.ndim == 1 and 0 <= plddt.min() and plddt.max() <= 100
    assert pae.ndim == 2 and pae.shape[0] == pae.shape[1] and pae.min() >= 0
    print("  OK: pLDDT per-residue in [0,100]; PAE square, non-negative Å.\n")


def check_dataset_item() -> None:
    ds = U.build_dataset(dummy=True)
    print(f"=== Dataset (--dummy): {len(ds)} items ===")
    ex = ds[0]
    n_mhc, n_pep = ex["n_mhc"], ex["n_pep"]
    n = int(ex["aatype"].shape[0])
    assert n == n_mhc + n_pep
    for k, shape in [("aatype", (n,)), ("residue_index", (n,)), ("seq_mask", (n,)),
                     ("anchor", (n,)), ("teacher_ca", (n, 3)), ("segment_id", (n,)),
                     ("teacher_plddt", (n,)), ("teacher_pae", (n, n))]:
        assert tuple(ex[k].shape) == shape, f"{k}: {tuple(ex[k].shape)} != {shape}"
    assert torch.all(ex["seq_mask"] == 1.0), "single item should be fully unmasked"
    assert 0 <= float(ex["teacher_plddt"].min()) and \
           float(ex["teacher_plddt"].max()) <= 100
    print(f"  item 0 id={ex['id']}  N={n} (n_mhc={n_mhc}, n_pep={n_pep})")
    print("  per-item shapes + ranges OK.\n")


def check_collate_padding() -> None:
    ds = U.build_dataset(dummy=True)
    # pick two different lengths (a 9-mer N=194 and an 8-mer N=193)
    items = [ds[i] for i in range(len(ds))]
    by_len = {int(e["aatype"].shape[0]): e for e in items}
    a, b = by_len[max(by_len)], by_len[min(by_len)]
    batch = U.collate_with_teacher([a, b])
    bsz, max_n = batch["aatype"].shape
    n_a = int(a["aatype"].shape[0])
    n_b = int(b["aatype"].shape[0])
    print(f"=== collate_with_teacher batch ===  B={bsz} max_N={max_n} "
          f"lengths={batch['length'].tolist()}")
    assert batch["teacher_plddt"].shape == (bsz, max_n)
    assert batch["teacher_pae"].shape == (bsz, max_n, max_n)
    # padding correctness on the shorter example (index 1)
    assert torch.all(batch["seq_mask"][1, n_b:] == 0), "pad seq_mask must be 0"
    assert torch.all(batch["teacher_plddt"][1, n_b:] == 0), "pad plddt must be 0"
    assert torch.all(batch["teacher_pae"][1, n_b:, :] == 0) and \
           torch.all(batch["teacher_pae"][1, :, n_b:] == 0), "pad pae must be 0"
    # real region preserved
    assert torch.allclose(batch["teacher_plddt"][1, :n_b], b["teacher_plddt"])
    assert torch.allclose(batch["teacher_pae"][1, :n_b, :n_b], b["teacher_pae"])
    assert torch.allclose(batch["teacher_pae"][0, :n_a, :n_a], a["teacher_pae"])
    print("  teacher pLDDT/PAE padded with 0 on masked positions; real region "
          "preserved. OK.\n")


def check_split_builder() -> None:
    print("=== split builder (real-mode id partitions) ===")
    import pandas as pd
    n_pool = len(pd.read_csv(U.POOL_CSV))
    for scheme in ("two_axis", "hla_only"):
        sp = U.read_split_ids(scheme, fold=1)
        tr, va, te = (set(sp["train"]), set(sp["val"]), set(sp["test"]))
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), \
            f"{scheme}: splits overlap"
        print(f"  {scheme} fold1: pool={n_pool:,}  train={len(tr):,}  "
              f"val={len(va):,}  test={len(te):,}  (disjoint OK)")
    print()


def main() -> None:
    inspect_one_dir()
    check_dataset_item()
    check_collate_padding()
    check_split_builder()
    print("GATE 2 PASSED: teacher files, Dataset item, collate padding, splits.")


if __name__ == "__main__":
    main()
