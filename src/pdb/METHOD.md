# PMGen-v2 distillation — data pipeline & sanity (src/pdb)

Coordinate-distillation data path for replacing AF2's Evoformer with a small
encoder while reusing the **frozen** AF2 structure module + pLDDT/PAE heads
(`src/afbuild/`). Env: `pmgen2` (`~/miniforge3/envs/pmgen2/bin/python`).

## What was done
- **Inspected** the teacher test set (`data/test/`, 15 class-I examples). Each PDB
  is a single chain "A", MHC-first then peptide, split by a ~200 residue-number
  gap (here **185→386**, +201). `inputs.tsv` cols: `peptide, mhc_seq, anchors,
  mhc_type, id`; `id`↔PDB 1:1. Teacher `plddt`/`pae` npy are post-padded → slice
  `[:N]` / `[:N,:N]`, `N = n_mhc + n_pep`.
- **`pdb/parse.py`** — `parse_example()` → tensors `aatype, residue_index,
  seq_mask, anchor, teacher_ca, segment_id, n_mhc, n_pep`. Order MHC→peptide;
  `aatype` from sequence via `residue_constants.restype_order`; `residue_index`
  keeps the gap; `anchor` 1s at `n_mhc+(p-1)` (1-indexed peptide); per-segment
  length+identity verified (raises on mismatch); class II guarded. Plus
  `collate_fn` (pad to max-N, `seq_mask=0`, `segment_id=-1`).
- **`pdb/check_parse.py`** — asserts lengths, the single boundary gap at `n_mhc`,
  anchors in peptide only, monotone `segment_id`. **All 15 files pass.**
- **`pdb/overfit_one.py`** — throwaway encoder (embeddings + sinusoidal PE → `s`;
  outer-concat MLP → `z`) + frozen stack + differentiable Kabsch superposed-Cα
  `smooth_l1` loss; trains encoder only.

## Results
- Parser/sanity: 15/15 OK.
- Overfit one example (frozen stack, 2000 steps): Cα-RMSD **17.6 → 0.6–1.2 Å**
  (3GSO_0_0 1.21 Å, 6UK4_0_0 0.59 Å) → ordering/indexing/masking/loss validated.
- Kabsch unit-tested (exact recovery). Early high plateau was just LR (L1 regime),
  not a bug; defaults set to lr 4e-3 / 2000 steps / cosine decay.
