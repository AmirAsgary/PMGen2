# model_multimer_1

A slim two-head model on top of the **frozen AlphaFold-Multimer** structure module +
pLDDT head. Single-sequence (no MSA), no templates, anchors via per-residue one-hot +
per-chain `residue_index`/`asym_id`. Trained confidence-first, then confidence-only.

## Pipeline
```
head-1  seq -> AF-multimer InputEmbedder (per-chain residue_index, asym/entity/sym,
        single-row msa_feat[49]) -> s_seq[N,256], z_seq[N,N,128]
head-2  MHC backbone (+Gaussian noise) -> distogram(22) + relative-orientation(3) =
        25-d pair geometry ; 1 multimer IPA -> 10-d per-residue summary.
        summary -> mean-pool -> single ; the 25-d geometry -> injected into the PAIR
anchor  2-d one-hot (peptide anchor) -> single & pair
trunk   project to D=64 (single AND pair). n x block, each:
          PAIR  : OPM(single->pair) + TriMul-out + TriMul-in + PairTransition
                  (this is what actually refines the pair)
          SINGLE: 2 multimer-IPA (4 heads; pair+geom -> single) + self-attention
        frames = noised-MHC + peptide-identity (fixed geometric context).
out     WIDEN 64 -> {single 384 -> plddt_proj -> FROZEN pLDDT head (confidence);
        (single 384, pair 128) -> FROZEN multimer StructureModule -> Ca/frames (FAPE)}
```
Internal working width is **D=64** (single & pair), IPAs at **4 heads**; dims are only
widened to the frozen-module sizes (384/128) at the final projections. That makes the
trainable stack **~0.44 M params** (embedder + head-2 + trunk + projections; SM &
pLDDT head frozen and excluded). Because 64/4-head ≠ AF-multimer's 384/12-head IPA,
the IPAs train **from scratch** (only the embedder / SM / pLDDT head are pretrained).
`forward -> (ca, plddt_logits, zeros_pae, frames)` = the `DistillModel` contract, so it
reuses model_1's `DistillLoss` (`lambda_pae=0`) + `train_one_epoch` + DDP.

## Run order (cluster)
```bash
# 0. new dataset -> H5  (once)
tar -xzf data/hasmig_mhcs/burial_score_output.tar.gz -C data/hasmig_mhcs/
sbatch src/model_multimer_1/preprocess_hasmig.sbatch
python src/model_multimer_1/preprocess_hasmig.py --merge \
    --out-dir data/processed/h5_store_hasmig --csv x --zip-dir x

# 0b. burial+pLDDT for the OLD store (for the stage-1 filter), if not done:
sbatch src/visualization/data_exploration.sbatch          # -> outputs/data_exploration/per_structure.csv

# 1. multimer weights (once)
bash src/model_multimer_1/download_multimer_weights.sh
$PY src/model_multimer_1/extract_multimer_weights.py       # -> input_embedder_mm/sm_mm/plddt_mm .pt

# 2. SMOKE TEST FIRST (validates the openfold wiring on one synthetic pMHC)
$PY src/model_multimer_1/model.py                          # prints "OK shapes ..."

# 3. MULTI-GPU SMOKE (10 random structures, 2 epochs, 2 GPUs) — do this first
sbatch src/model_multimer_1/smoke_mm1.sbatch     # confirms DDP train + sharded val

# 4. train (three stages)
STAGE=1 sbatch src/model_multimer_1/train_mm1.sbatch
STAGE=2 RESUME=checkpoints_mm1/mm1_stage1/last.pt sbatch src/model_multimer_1/train_mm1.sbatch
STAGE=3 RESUME=checkpoints_mm1/mm1_stage2/last.pt sbatch src/model_multimer_1/train_mm1.sbatch
```

## Multi-GPU (train + validation)
Training shards the epoch loader across ranks (`make_epoch_loader(rank, world)`).
**Validation is also distributed**: each rank evaluates an equal, disjoint shard and
the loss/metric terms are all-reduced example-weighted, so rank 0 logs the true global
means (no more rank-0-only eval that made the other ranks idle). `smoke_mm1.sbatch`
exercises both on 2×A100 with 10 random structures for 2 epochs — run it before a full
job. `--max-train N` now takes a **random** N (seeded by `--seed`), not the first N.

## Defaults worth knowing
- `--n-trunk 3` (was 1) — local overfit showed the trunk is the capacity lever; n-trunk 3
  reached ~0.05 Å peptide-RMSD on a 20-structure overfit vs 0.55 Å at n-trunk 1.
- `--mhc-noise 0.1` (was 0.5) — 0.5 over-regularized (capped even train-set fit at ~1.7 Å);
  0.1 keeps robustness to imperfect input MHC without blocking a tight fit. Noise is only
  applied while `.training` (eval/validation always uses the clean MHC).
- `--unfreeze-sm-pct P --unfreeze-sm-at E` — optionally fine-tune the last P% of the frozen
  StructureModule from epoch E (rebuilds opt/sched + re-wraps DDP). Tested: at 10%/ep10 it
  barely helped (0.52 vs 0.55 Å), so the SM freeze is a fine default.

## Three-stage schedule
- **Stage 1** — trunk **+ SM** trainable (pLDDT head frozen). Structure-first: pLDDT
  weight 0 on epoch 1, then `--plddt-w` (0.01). Trained on the high-confidence subset.
- **Stage 2** — ALL structures, SM stays trainable, pLDDT weight 0.01 (`lam=(1,0.01,0)`).
- **Stage 3** — freeze the **whole model except `plddt_proj`** (pLDDT head always frozen);
  `s` is detached into the pLDDT path so confidence learning can't move the structure.
  Confidence-only, pLDDT weight 1.0 (`--stage3-plddt-w`). Resume from stage-2.

`set_stage(stage)` in `model.py` applies the freezing; `train.py` picks `lam` per stage.

## Confidence / burial filter (stage 1 only)
`burial >= 0.65 AND peptide pLDDT > 0.70`. Old store: `burial_score` + `mean_peptide_plddt`
(0–100, so threshold ×100) from `outputs/data_exploration/per_structure.csv`. New store:
`docking_score` (= burial) + `pep_mean_plddt` (0–1) from its `index.csv`. Stages 2/3 use
ALL structures. `--no-filter` disables it; `--max-train N` caps the train set (overfit).

## Status: validated end-to-end locally
Smoke + a 20-structure overfit were run locally (pmgen2 env, one GPU, bf16 AMP): the
full path (real hasmig zips → H5 → dataset → `InputEmbedderMultimer` → head-2 →
trunk → frozen multimer SM → FAPE → backward → DDP) runs clean at ~11 it/s. Overfit
drove FAPE from ~2.46 (init) to ~0.26 and peptide Cα-RMSD to ~1.6 Å, confirming the
Rigid3Array frames, `StructureModule(is_multimer=True)` I/O, and `*_mm.pt` weight keys
are all wired correctly. Run the `--smoke` (step 2) before any full training anyway.
