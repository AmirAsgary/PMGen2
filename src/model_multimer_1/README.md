# model_multimer_1

A slim encoder on top of the **frozen AlphaFold-Multimer** structure module + pLDDT
head. Single-sequence (no MSA), no templates; anchors via a per-residue one-hot + the
per-chain `residue_index`/`asym_id` the embedder already understands. The MHC backbone
is given as an INPUT (head-2), so the task is "given noisy MHC + sequences + anchors,
dock the peptide", not de-novo folding. Trained in three stages: structure-first, then
a broader-data structure+confidence stage, then confidence-only.

## Pipeline
```
head-1  seq -> AF-multimer InputEmbedder (per-chain residue_index, asym/entity/sym,
        single-row msa_feat[49]) -> s_seq[N,256], z_seq[N,N,128]
head-2  MHC backbone (+Gaussian noise, train only) -> distogram(22) + relative-
        orientation(3) = 25-d pair geometry ; 1 multimer IPA -> 10-d per-residue summary.
        summary -> mean-pool -> single ; the 25-d geometry -> injected into the PAIR
anchor  2-d one-hot (peptide anchor) -> single & pair
trunk   project to D=64 (single AND pair). n_trunk x block, each:
          PAIR  : OPM(single->pair) + TriMul-out + TriMul-in + PairTransition
          SINGLE: 2 multimer-IPA (4 heads; pair+geom -> single) + self-attention
        frames = noised-MHC + peptide-identity (fixed geometric context).
out     WIDEN 64 -> {single 384 -> plddt_proj -> FROZEN pLDDT head (confidence);
        (single 384, pair 128) -> FROZEN multimer StructureModule -> Ca/frames (FAPE)}
```
Internal working width is **D=64** (single & pair), IPAs at **4 heads**; dims are only
widened to the frozen-module sizes (384/128) at the final projections. `forward ->
(ca, plddt_logits, zeros_pae, frames)` = the `DistillModel` contract, so it reuses
model_1's `DistillLoss` (`lambda_pae=0`) + `train_one_epoch` + DDP + metric logging.
AF-multimer runs in **bf16** (fp16 overflows the Evoformer/SM).

## What is frozen vs trainable
Three modules load pretrained AF-multimer weights (`*_mm.pt`): `embedder`, `sm`, `plddt`.
"Pretrained" ≠ "frozen": the **embedder is fine-tuned**; the **StructureModule and pLDDT
head are ALWAYS frozen, in every stage** (and forced to `.eval()`, so they never see
dropout/SM stochasticity). The trunk IPAs run at 64/4-head ≠ AF's 384/12-head, so they
**train from scratch**. Only the encoder path is ever trained.

| module | params | Stage 1 | Stage 2 | Stage 3 |
|---|--:|:--:|:--:|:--:|
| embedder (head-1) | 33.5k | train | train | frozen |
| head-2 (MHC encoder) | 56.3k | train | train | frozen |
| s_proj / z_proj | 27.2k | train | train | frozen |
| sm_s / sm_z (widen→SM) | 33.3k | train | train | frozen |
| **plddt_proj** (widen→pLDDT) | 25.0k | train | train | **train** |
| trunk (n_trunk=3) | 784k | train | train | frozen |
| **sm** (StructureModule) | 2.02M | **frozen** | **frozen** | **frozen** |
| **plddt** (AF pLDDT head) | 73k | **frozen** | **frozen** | **frozen** |
| **trainable total** | | **959k** | **959k** | **25k** |

`set_stage(stage)` in `model.py` applies this. (The opt-in `--unfreeze-sm-pct P
--unfreeze-sm-at E` can fine-tune the last P% of the SM from epoch E — tested at 10%/ep10,
it barely helped (0.52 vs 0.55 Å overfit), so it is OFF by default and the SM stays frozen.)

## Three-stage schedule
- **Stage 1 (structure-first)** — encoder + trunk + projections trainable; SM + pLDDT
  frozen. Trained on the confidence-**filtered** subset. pLDDT weight 0 on epoch 1, then
  `--plddt-w` (0.01). hasmig at full weight. `lam=(1, plddt-w, 0)`.
- **Stage 2 (broader data)** — SAME trainable set as stage 1 (SM stays frozen); pLDDT
  weight 0.01; hasmig down-weighted. Differs from stage 1 only by the DATA/loss regime —
  see the two approaches below. `lam=(1, 0.01, 0)`.
- **Stage 3 (confidence-only)** — freeze the whole model except `plddt_proj`; the forward
  detaches `s` before the pLDDT path so the structure can't move. pLDDT-only loss at
  `--stage3-plddt-w` (default 1.0). `lam=(0, stage3-plddt-w, 0)`.

Each stage is a separate job; stage k+1 resumes stage k's checkpoint (loads weights,
fresh optimizer/scheduler because the stored `stage` differs).

### Stage 2 — two alternative experiments (each a separate 4-GPU job)
Both resume from a frozen copy of stage-1's **LAST** checkpoint — `stage2_mm1.sbatch`
copies `mm1_stage1/last.pt → checkpoints_mm1/pretrained_stage1_last.pt` once, so A and B
start from the identical init. Both: SM frozen, pLDDT weight 0.01, hasmig down-weighted
0.1, 2 epochs. Submit as two independent `gpu:a100:4` jobs:
The approach is the **first positional argument** (`A`/`B`) — pass it as an argument, not
`APPROACH=B sbatch …`, because SLURM does not reliably propagate that env var to the batch
job (it silently falls back to A and both jobs clobber `mm1_stage2_A/`).
```bash
cp checkpoints_mm1/mm1_stage1/last.pt checkpoints_mm1/pretrained_stage1_last.pt   # once
sbatch src/model_multimer_1/stage2_mm1.sbatch A   # -> checkpoints_mm1/mm1_stage2_A
sbatch src/model_multimer_1/stage2_mm1.sbatch B   # -> checkpoints_mm1/mm1_stage2_B
```
- **A (`--force-filter --filter-val`)** — keep training AND validating on the confidence-
  **filtered** structures (same filter as stage 1). Clean, in-distribution; ignores the
  low-quality tail.
- **B (`--struct-quality-weight`)** — FULL dataset, **no** filter. Each structure's
  **structural (FAPE)** loss is scaled by a quality weight
  `w_n = w_min + (1-w_min)·q_plddt(pₙ)·q_burial(bₙ)` (`--w-min`, default 0.05;
  `q_burial = min(b/0.65, 1)`; `q_plddt` a 6-step ramp: ≥0.9→1.0, ≥0.8→0.85, ≥0.7→0.70,
  ≥0.6→0.50, ≥0.5→0.25, else 0.10). Low-quality structures still contribute weakly instead
  of being dropped. **Only FAPE is quality-weighted** — the pLDDT CE stays at full weight
  on ALL structures, so the model still learns to predict LOW confidence for the
  low-quality/non-binder tail. Validation on ALL structures.

A and B validate on **different** sets, so their absolute `val` numbers aren't head-to-head;
compare each against its own train, or eval both final checkpoints on one common set after.

## Data split (train vs val / test)
- **Train** = OLD-store `two_axis` **train** ids (confidence-filtered in stage 1) **+ ALL
  hasmig ids**. Choose the split with `--scheme two_axis --fold {1..5}` (default two_axis/1;
  reuses model_1's `read_split_ids`).
- **Val / test** = OLD-store `two_axis` **val / test** ids **only** (unfiltered). The HLA
  two-axis split lives only on the old data, so **hasmig is never validated/tested on** — it
  is a training-only, low-diversity augmentation. Test (`test.csv`) is a separate eval, not
  run in the training loop.
- **`val` vs `val-matched`.** Stage 1 trains on a filtered subset, but ~92% of the held-out
  val fold is BELOW that filter (median pep-pLDDT ≈ 50), so `val` loss sits well above
  `train` purely from the train/val distribution shift — **not** overfitting. Each epoch
  therefore also logs **`val-matched`**: the val subset passing the SAME filter as train.
  `train ≈ val-matched` ⇒ no overfitting (the `val` gap is data quality); `val-matched ≫
  train` ⇒ genuine overfitting. Logged as split `val_matched` in `metrics.csv`. (Skipped in
  Approach A, where `val` is already the filtered set, and in stages 2/3 without a filter.)

## Losses & per-example weighting
Loss = `DistillLoss` = λ_fape·FAPE + λ_plddt·CE(pLDDT) (λ_pae=0). `peptide_weight=5.0`
up-weights peptide residues/pairs so the small peptide isn't drowned by the MHC. Two
composable **per-example** weights, both carried through `collate_with_teacher` (each
defaults to 1.0 → no-op) and applied inside `DistillLoss`:
- **`sample_weight`** — source weight; scales **FAPE + pLDDT CE**. hasmig gets
  `--hasmig-weight` (default **0.1**) in stages 2/3 (high quality but very low MHC-sequence
  diversity → curbs over-fit); old data and stage 1 stay 1.0.
- **`struct_weight`** — the quality `w_n`; scales **FAPE only** (Approach B), so the pLDDT
  CE still learns confidence on every structure.

**Validation uses the identical weighting as training** (the same annotator wraps train,
val, and val-matched), so reported val and train totals are directly comparable.

## Dropout
`DROPOUT = 0.1` on the **trainable** layers only (head-2 output, the two trunk-input
projections, and inside every `TrunkBlock`: each pair-update branch, each IPA residual, the
single self-attention). Frozen SM / pLDDT are held in `.eval()` so they never see dropout;
validation (`model.eval()`) turns it off automatically.

## Confidence / burial filter
`burial >= --burial-min (0.65) AND peptide pLDDT > --plddt-min (0.70)`. Old store:
`burial_score` + `mean_peptide_plddt` (0–100, ×100 threshold) from
`outputs/data_exploration/per_structure.csv`. hasmig: `docking_score` (= burial) +
`pep_mean_plddt` (0–1) from its `index.csv`. Applied in **stage 1** (train) by default, or
in any stage with `--force-filter` (Approach A); `--filter-val` filters val too;
`--no-filter` disables it. `--max-train N` caps the train set to a **random** N (overfit).

## Multi-GPU (train + validation)
Training shards the epoch loader across ranks (`make_epoch_loader(rank, world)`).
**Validation is also distributed**: each rank evaluates an equal, disjoint shard and the
loss/metric terms are all-reduced example-weighted, so rank 0 logs the true global means
(no rank-0-only eval that idled the other ranks). `smoke_mm1.sbatch` exercises both on
2×A100 (10 random structures, 2 epochs) — run it before a full job.

## Key defaults
- `--n-trunk 3` — the trunk is the capacity lever: n_trunk 3 reached ~0.05 Å peptide-RMSD on
  a 20-structure overfit vs 0.55 Å at n_trunk 1.
- `--mhc-noise 0.1` — regularizes against imperfect input MHC; 0.5 over-regularized (capped
  even train-set fit at ~1.7 Å). Applied only while `.training`; eval/val use clean MHC.
- `--hasmig-weight 0.1`, `--w-min 0.05`, `--peptide-weight 5.0`, `--plddt-w 0.01`.

## Run order (cluster)
```bash
# 0. new dataset -> H5  (once)
tar -xzf data/hasmig_mhcs/burial_score_output.tar.gz -C data/hasmig_mhcs/
sbatch src/model_multimer_1/preprocess_hasmig.sbatch
python src/model_multimer_1/preprocess_hasmig.py --merge \
    --out-dir data/processed/h5_store_hasmig --csv x --zip-dir x

# 0b. burial+pLDDT for the OLD store (stage-1 filter), if not done:
sbatch src/visualization/data_exploration.sbatch      # -> outputs/data_exploration/per_structure.csv

# 1. multimer weights (once)
bash src/model_multimer_1/download_multimer_weights.sh
$PY src/model_multimer_1/extract_multimer_weights.py  # -> input_embedder_mm/sm_mm/plddt_mm .pt

# 2. openfold-wiring smoke (one synthetic pMHC), then the multi-GPU smoke
$PY src/model_multimer_1/model.py                     # "OK forward ... OK loss ..."
sbatch src/model_multimer_1/smoke_mm1.sbatch          # DDP train + sharded val, 10 structs

# 3. stage 1   (STAGE is the first POSITIONAL ARG — env vars are not reliably
#               propagated to SLURM batch jobs, so pass it as an argument)
sbatch src/model_multimer_1/train_mm1.sbatch 1

# 4. stage 2 — two experiments, each its own 4-GPU job (approach = first arg: A or B)
cp checkpoints_mm1/mm1_stage1/last.pt checkpoints_mm1/pretrained_stage1_last.pt
sbatch src/model_multimer_1/stage2_mm1.sbatch A
sbatch src/model_multimer_1/stage2_mm1.sbatch B

# 5. stage 3 (confidence-only), resuming the chosen stage-2 run:  args = STAGE RESUME
sbatch src/model_multimer_1/train_mm1.sbatch 3 checkpoints_mm1/mm1_stage2_A/last.pt
```

## Status: validated end-to-end
Local (pmgen2, 1 GPU, bf16) smoke + overfit: the full path (real hasmig zips → H5 → dataset
→ `InputEmbedderMultimer` → head-2 → trunk → frozen multimer SM → FAPE → backward → DDP) runs
clean. A clean 20-structure overfit at n_trunk 3 drove peptide Cα-RMSD to **~0.05 Å**,
confirming the Rigid3Array frames, `StructureModule(is_multimer=True)` I/O, and `*_mm.pt`
weight keys are wired correctly and that the frozen-SM design has ample capacity. The 2-GPU
`smoke_mm1.sbatch` passed on Raven (sharded DDP train + all-reduced val). Run the smokes
(step 2) before any full training.
