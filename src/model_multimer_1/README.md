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

# 3. train
STAGE=1 sbatch src/model_multimer_1/train_mm1.sbatch
STAGE=2 RESUME=checkpoints_mm1/mm1_stage1/last.pt sbatch src/model_multimer_1/train_mm1.sbatch
```

## Confidence / burial filter (stage 1)
`burial >= 0.65 AND peptide pLDDT > 0.70`. Old store: `burial_score` + `mean_peptide_plddt`
(0–100, so threshold ×100) from `outputs/data_exploration/per_structure.csv`. New store:
`docking_score` (= burial) + `pep_mean_plddt` (0–1) from its `index.csv`. Stage 2 uses
ALL structures (no filter).

## IMPORTANT: first run is a wiring-validation pass
This wires several OpenFold-multimer internals (Rigid3Array frames for the multimer
IPA, `StructureModule(is_multimer=True)` I/O, `InputEmbedderMultimer`). It compiles and
the feature layout is confirmed, but it has **not** been run end-to-end (no local
GPU/openfold). **Run the `--smoke` (step 2) first**; the most likely spots to need a fix
are (a) `_frames_from_bb` Rigid→Rigid3Array construction, (b) the multimer IPA/SM tensor
dims, (c) the frozen `sm_mm.pt`/`plddt_mm.pt` key names from extraction. Report the smoke
output/traceback and I'll fix the seam.
