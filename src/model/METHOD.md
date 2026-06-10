# PMGen-v2 encoder + distillation — architecture & verification

Goal: replace AlphaFold2's Evoformer with a small **trainable encoder** while
reusing AF2's **frozen** structure module + pLDDT/PAE heads. The encoder maps
sequence/anchor features to `s:[B,N,384]` and `z:[B,N,N,128]`; the frozen stack
turns them into coordinates + pLDDT/PAE logits. We train **only** the encoder.

Env: `pmgen2` (`~/miniforge3/envs/pmgen2/bin/python`). Code: `src/model/`
(`utils.py` = bulk, `train.py` = entrypoint, `*_test.py` = gates). Reuses
`load_frozen_fold` (src/afbuild), `parse_example`/`collate_fn` (src/pdb), and
OpenFold blocks — nothing reimplemented.

## Architecture (PART 1 — `utils.py`)

**Featurizer.** Embeds `aatype`(21), `segment_id`(0=MHC/1=peptide, +pad slot),
`anchor`(0/1) → `s0:[B,N,d_s]`. Pair init `z0:[B,N,N,d_z]` = outer combo of the
single tokens (`left_i + right_j`) plus an AF2 relative-position one-hot of
`residue_index` with the offset **clipped to ±32**, so the ~200 MHC↔peptide
numbering gap saturates to "far".

**Encoder block** (`depth` blocks, default 1). Order is **z first, then s**:
1. *Pair update* — variant-selected triangular ops, each with its own LayerNorm
   and an external residual; pair mask = `outer(seq_mask)`.
2. *Single update* — self-attention on `s` **biased by `z`** (a `Linear(d_z→heads)`
   projection of LayerNorm(z) gives a per-head bias; a key-mask bias hides padded
   columns) via OpenFold `Attention`, then a transition MLP. LayerNorm+residual
   around each sub-op.

Final LayerNorm + `Linear` project to the frozen widths exactly: `s:[B,N,384]`,
`z:[B,N,N,128]`. Everything is masked at every cross-position op.

**7 variants (pair update only), all implemented:**
| # | pair update |
|---|---|
| 1 | TriMul Incoming + Outgoing |
| 2 | TriAttn Starting + Ending |
| 3 | TriMul Incoming only |
| 4 | TriMul Outgoing only |
| 5 | TriAttn Starting only |
| 6 | TriAttn Ending only |
| 7 | FULL: TriMulOut → TriMulIn → TriAttnStart → TriAttnEnd → pair transition |

**`DistillModel`** = `DistillEncoder` + `load_frozen_fold`. `train()` keeps the
frozen stack in eval; `trainable_parameters()` exposes encoder-only params; all
frozen params have `requires_grad=False`.

## Data layer (PART 2 — `utils.py`)

- **`DistillDataset`**: per row → `parse_example(...)` + sliced teacher arrays.
  Teacher npy are post-padded → sliced `plddt[:N]` (per-residue, 0–100) and
  `pae[:N,:N]` (Ångström), `N = n_mhc + n_pep`. `find_teacher_files` resolves the
  PDB, `*_plddt.npy`, and `*_predicted_aligned_error.npy` (PAE, not the pTM npy).
- **`collate_with_teacher`**: reuses `collate_fn` for the parsed tensors, then
  pads teacher pLDDT `[B,N]` / PAE `[B,N,N]` with 0 on masked positions.
- **Split builder** `read_split_ids(scheme, fold)`: maps the processed split
  files' `row_idx` to base ids; `train = pool − fold val − test`; schemes
  `two_axis`, `hla_only`, folds 1–5.
- **Multiple anchors:** real-mode `build_dataset` draws one example per *anchor
  combination* (rows of `Multiple_Anchors_input_reduced.tsv`) and assigns each to
  the split of its **base id** (`base_id` strips the `_<idx>` anchor suffix), so
  **all anchor combinations of a (peptide, MHC) pair land in the same split**.
  Each anchor id has its own teacher dir (`af_root/<anchor_id>`).
- **`--dummy`** routes the same code path at `data/test/` (15 class-I examples).

## 3-term loss & metrics (PART 2 — `utils.py`)

`DistillLoss` = `λ_fape·FAPE + λ_plddt·CE(plddt, 50-bin) + λ_pae·CE(pae, 64-bin)`
(defaults 1.0 / 0.1 / 0.1). The CE terms backprop through the *frozen,
ungradiented* heads + SM into the encoder.
- **FAPE**: OpenFold `backbone_loss` on the SM frame trajectory vs GT backbone
  frames built from teacher N/Cα/C via `atom37_to_frames`+`get_backbone_frames`.
  **Unclamped** (`fape_clamp=None`) — the AF2 10 Å clamp zeroes the gradient when
  every error exceeds it, so a random init can't bootstrap.
- **pLDDT**: teacher pLDDT(0–100) → 50 uniform bins; masked CE vs `[B,N,50]`.
- **PAE**: teacher PAE(Å) → 64 bins with breakpoints from `loss.tm`
  (`max_bin=31`, not hardcoded); masked CE vs `[B,N,N,64]`.
- Eval metrics: Kabsch Cα-RMSD, pLDDT Spearman, PAE MAE.

Two fixes were required for the structure to fold (both verified by overfit):
1. **Output init**: the encoder's `s_out`/`z_out` are the *primary* SM inputs, so
   they must use non-zero init — `init="final"` (zero) starts the SM from a black
   hole with no gradient.
2. **Unclamped FAPE** (above).

## Verification gates (run in order, each must pass)

**GATE 1 — `encoder_test.py` (PASSED).** All 7 variants, on a real example and a
padded batch of 2:
- Shapes: encoder `s[B,N,384]`/`z[B,N,N,128]`; full pass `ca[B,N,3]`,
  `plddt[B,N,50]`, `pae[B,N,N,64]`.
- Gradient isolation: encoder params get grad; every frozen param has
  `requires_grad=False` and `.grad is None`.
- **No padding leak**: perturbing padded slots leaves all real outputs (encoder
  s/z and frozen ca) **bit-identical** (`leak_dev = 0.0` for every variant).
- Cost (B=2, N=194, CUDA): params 1.45–1.80 M, fwd 35–49 ms, peak 1.4–3.1 GB
  (variant 7 the heaviest).

**GATE 2 — `data_test.py` (PASSED).**
- Teacher files resolved; **pLDDT per-residue ∈ [58.6, 98.9]**, **PAE N×N Å ∈
  [0.25, 30.14]** (diagonal ≈ 0.25 Å).
- Dataset item: all tensor shapes (incl. `teacher_plddt[N]`, `teacher_pae[N,N]`)
  and masks correct.
- `collate_with_teacher`: teacher arrays padded with 0 on masked positions, real
  region preserved.
- Split builder disjoint: two_axis f1 train 256,053 / val 9,257 / test 2,241;
  hla_only f1 train 191,892 / val 48,681 / test 26,978 (pool 267,551).

**GATE 3 — `overfit_test.py` (PASSED).** `--dummy` overfit one example (3GSO_0_0,
variant 7, full 3-term loss, 4000 steps, lr 1e-2): FAPE **2.90→0.68**, Cα-RMSD
**19.94→0.95 Å** (<2 Å), pLDDT Spearman **−0.00→0.92**, PAE MAE **12.1→1.50 Å**.
(The CE terms train `z` directly and dominate early, delaying folding — hence the
larger step budget; FAPE-only folds by ~1700 steps.) GATE 1 re-checked after the
init/forward changes: still passes (leak 0.0).

**GATE 4 — `train_test.py` (PASSED).** `--dummy` short run over all 15 examples
(variant 7, bs 3, 12 epochs): padded-teacher masking exact (perturbing padded
pLDDT/PAE/backbone leaves the loss unchanged, Δ=0.0) and **train loss
3.331→2.664** (driven mostly by the pLDDT/PAE CE terms; FAPE/structure needs many
more steps to fold, cf. GATE 3). Per-epoch per-term + val-metric table printed.

## Training (PART 2 — `utils.py` + `train.py`)
`run_training(...)` (heavy lifting in utils) builds train/val datasets + loaders
(`make_dataloader` with `collate_with_teacher`), a `DistillModel` + `DistillLoss`,
**AdamW over encoder params only**, cosine LR, optional AMP, grad clip; logs
per-epoch per-term train losses + val Cα-RMSD / pLDDT-Spearman / PAE-MAE. Single-
GPU default, DDP-ready (sampler hook + external DDP wrap). `train.py` is a thin
argparse CLI (`--variant --scheme --fold --dummy --af-root --epochs --bs --lr
--lambdas --amp --grad-clip ...`).

**All four gates pass.** The data layer is real-mode-complete (multiple-anchor
base-id splits); it just needs teacher PMGen predictions under `--af-root` for the
267k/885k anchor pool (only the 15 dummy examples have teachers today).

## Streamable HDF5 store (one-time preprocessing)

For full-scale training we don't re-parse PDBs every epoch. `preprocess.py`
(+ `preprocess_chunks` in `utils.py`) turns the chunked PMGen outputs into a
streamable store:
- Input layout: `<chunks-dir>/chunk_*/chunk.tsv` + per-id PMGen outputs under
  `chunk_*/<output-link>/<alphafold-subdir>/<id>/`.
- Output: **one HDF5 shard per chunk** (`<out>/chunk_*.h5`, one group per anchor
  id) + a merged `index.csv` (`id, base_id, shard, n_mhc, n_pep, mhc_type,
  cluster ids`). Each chunk is independent → **parallel jobs** on disjoint
  `--chunks`, **resumable** (existing shards skipped). Compact dtypes
  (aatype u1, residue_index i2, PAE f16+gzip, coords f32) ≈ **86 KB/example**
  (~74 GB for 885k; PAE dominates).
- **Splits = id lists, not duplicated data**: `H5DistillDataset` reads by anchor
  id; `build_h5_dataset(scheme, fold, split)` selects anchor ids whose base id is
  in that partition (via `read_split_ids`), so all anchor combos of a pair stay
  together and each example's heavy data is stored once.
- Training uses it via `train.py --h5-dir <out> --scheme … --fold …` (or
  `--dummy`); `run_training(h5_dir=…)` wires the H5 datasets in. `h5py` handles
  are opened lazily (fork-safe for DataLoader workers).

**GATE 5 — `h5_test.py` (PASSED).** Builds a fake chunk from the 15 dummy examples,
preprocesses → shard + `index.csv`, and verifies: round-trip equality
(aatype/residue_index/anchor/segment + ca/bb **exact**, pLDDT/PAE within float16
tol), 3-term loss on an H5 batch, and a short training off the store decreasing
the loss.
