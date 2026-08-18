# model_multimer_1

> ## ⚠ CURRENT STATE (2026-08-18): there is NO usable checkpoint
> **`checkpoints_mm1/mm1_stage1_sc/last.pt` is 292/292 NaN tensors** (plus 584 NaN Adam
> states). `checkpoints_mm1_bk/*` are all pre-leak-fix and unusable per the warning below.
> Nothing in this directory can currently be evaluated or deployed.
>
> The only leak-free run ever attempted (job `28811055`, 13 Jul, logs/mm1/28811055.out):
>
> | steps | what happened |
> |---|---|
> | 1 – 24,800 | healthy — FAPE 2.14 → 0.11, chi 0.36 → 0.059, Cα-RMSD ~0.25 Å |
> | 24,850 – 25,050 | exponential runaway (0.20 → 0.26 → 0.33 → 0.49 → 0.98 → 2.15) |
> | 25,050 – 33,000 | **dead** at random-init loss, Cα-RMSD 17–20 Å, for 8,000 steps |
> | 33,050 | NaN; cancelled at 15:04 |
>
> It reached **35% of epoch 1 of 6** and therefore **never ran a validation pass**. Every
> held-out number quoted anywhere in this README comes either from the leaky checkpoints
> or from 20–40 structure probes. **No honest generalisation number exists for this model.**
>
> The stability guards added afterwards (`93a9f74`, `29f5bbc`, `00e4e79` — non-finite step
> skip, NaN-safe save, `best.pt`, divergence abort, `|g|` telemetry, LR warmup,
> `--trunk-fp32`) were committed but **never exercised by a real job**: the last mm1 log
> predates them. Treat them as untested until a run confirms otherwise.
>
> Restart path: `sbatch src/model_multimer_1/run_stage1_short.sh` (≈1 h, 4 GPUs) — proves
> the harness trains/validates/checkpoints honestly and yields the first held-out number;
> then the full `run_stage1_sidechain.sh`.

> ## ⚠ FIXED BUG: side-chain losses escaped every weighting
> `sc_fape` and `chi_loss` were plain `.mean()`s over every residue of every example, so
> they bypassed **all three** weightings the rest of the loss applies:
> `--peptide-weight` (they were ~95% MHC — 181 residues vs a 9-mer — i.e. mostly
> re-scoring rotamers of a backbone handed in as INPUT), `--hasmig-weight`, and
> `--struct-quality-weight`. At the magnitudes actually logged,
> `0.5·sc_fape + 1.0·chi` is **~71% of the total loss**, so **Approach B's entire premise
> failed silently**: low-confidence structures kept full influence over most of the
> objective, and the "very important" peptide side chains were drowned out.
>
> Fixed in `DistillLoss` via `_sidechain_fape_per_example` (routes the per-residue weight
> through `compute_fape`'s `pair_mask`, which enters the normaliser, so it is a correct
> weighted mean) and `_supervised_chi_loss_per_example` (openfold's `supervised_chi_loss`
> reduces the batch to a **scalar** internally, so no per-example weight could reach it at
> all). Verified: unweighted path reproduces openfold to 4e-7 (sc-FAPE) and bit-exactly
> (chi, B=1); at `peptide_weight=5` the peptide's share of the side-chain signal rises
> 13% → 43% and both terms respond to `struct_weight` for the first time.
> `--unweighted-sidechain-losses` restores the old behaviour for A/B comparison.
>
> Also fixed: `evaluate()` ran **without `torch.no_grad()`**, building and retaining the
> full autograd graph through the frozen AF stack on every validation batch.


> ## ⚠ KNOWN BUG: the peptide pose leaks into the trunk (`pep_frames`)
> `_frames_from_bb` only forces the *padding* residues to the identity frame, so the
> **peptide's TRUE backbone frames** (from `teacher_bb`) are fed to the trunk IPAs —
> the model can read the answer instead of docking. The pipeline diagram below says
> "peptide-identity"; the code did not do that. Measured on the 15 `data/test` examples
> with the stage-2 checkpoints (peptide Cα-RMSD, superposed on MHC):
>
> | checkpoint | `pep_frames=teacher` (leaky) | `pep_frames=identity` (honest) |
> |---|---|---|
> | A_resume | **0.26 Å** | **7.65 Å** |
> | B_resume | **0.33 Å** | **9.69 Å** |
>
> Translating only the peptide's `teacher_bb` by 10 Å moves the predicted peptide by
> ~6.7 Å while the MHC stays put — the prediction follows the input. **All peptide-RMSD
> numbers reported so far (train ~0.26 Å, val ~0.28 Å) are contaminated**, and the model
> cannot be deployed (at inference you don't know the peptide backbone).
> **FIXED.** `--pep-frames identity` is now the **default**; `teacher` reproduces the old
> checkpoints and prints a loud warning. Every job runs `_leak_check` pre-flight.
> All pre-fix checkpoints are unusable (their trunk learned to read the answer) — retrain
> from stage 1: `sbatch src/model_multimer_1/run_stage1_sidechain.sh`.
>
> **`identity` is learnable** — 20-structure overfit (lr 5e-4, n_trunk 3, `--mhc-noise 0`,
> seed 0, 250 epochs, 0 NaN): FAPE 2.16 → 0.114, peptide-RMSD 10.9 → 0.73 Å. The leaky
> `teacher` run at the **same** budget reaches FAPE 0.079 / 0.71 Å — i.e. **the leak helps
> slightly, as it must** (more information). An earlier 60-epoch comparison appeared to show
> `identity` *beating* `teacher`; that was an artifact of the cosine schedule (`T_max` =
> `--epochs`, so LR hit 0 by epoch 60) — `teacher` starts slower because the IPA's
> zero-initialised output projection must first learn to exploit the frames. Do not read
> short runs. On a 20-structure overfit both simply memorise, so the leak buys little; its
> damage shows up on **held-out** data (0.26 Å reading the answer vs 7.6 Å without it).
> **LR matters: 2e-3 diverges to NaN by epoch 15; 5e-4 is stable.** Use lr ≤ 5e-4.
>
> `python src/model_multimer_1/model.py` now runs `_leak_check()` — it asserts on the
> *frames* (the sole channel by which `teacher_bb` reaches the peptide) that under
> `identity` they are exactly the identity and do not move when the peptide's `teacher_bb`
> is perturbed, plus a positive control that `teacher` does move. Asserting on the frames
> rather than the prediction is deliberate: OpenFold's IPA zero-inits its output projection,
> so on a fresh model the frames have **no** effect on the output and an end-to-end check
> would pass even with the leak present.

A slim encoder on top of the **frozen AlphaFold-Multimer** structure module + pLDDT
head. Single-sequence (no MSA), no templates; anchors via a per-residue one-hot + the
per-chain `residue_index`/`asym_id` the embedder already understands. The MHC backbone
is given as an INPUT (head-2), so the task is "given noisy MHC + sequences + anchors,
dock the peptide", not de-novo folding. Trained in three stages: structure-first, then
a broader-data structure+confidence stage, then confidence-only.

## ⚠ The multiple structures per complex are ANCHOR VARIANTS, not replicates

Store ids are `ALLELE_PEPTIDE_LENGTH_INDEX` (e.g. `HLAA01138_FTGRFPGYVV_10_3`). The
873,749 structures cover 264,009 distinct `ALLELE_PEPTIDE_LENGTH` complexes — about 3.3
structures each — and it is tempting to read those as AlphaFold replicates of one
prediction. **They are not.** PMGen hands AlphaFold a different **anchor** for each one,
and the structure changes accordingly. Measured over all 225,616 multi-structure
complexes: **100.0% have all-distinct anchors** (0.0% repeat one).

```
HLAA01138_FTGRFPGYVV_10_3   anchor 1;10   pep-pLDDT 47.1
HLAA01138_FTGRFPGYVV_10_5   anchor  2;9   pep-pLDDT 45.8      MHC differs by 0.19 A
HLAA01138_FTGRFPGYVV_10_7   anchor  3;9   pep-pLDDT 47.0      PEPTIDE by up to 5.30 A
HLAA01138_FTGRFPGYVV_10_9   anchor 4;10   pep-pLDDT 54.9
```

So the several-Angstrom spread between structures of one complex is **the training
signal, not noise**: `anchor` is a model INPUT, and learning anchor -> structure is the
task. Consequences, all of which are easy to get wrong:

* **Never deduplicate by `base_id`.** Collapsing a complex to one structure (medoid or
  otherwise) deletes exactly what the model is supposed to learn.
* **Inter-structure spread is NOT a teacher-noise estimate.** The store contains no
  same-anchor repeats at all, so AlphaFold's run-to-run noise cannot be measured from
  it, and the spread must not be used as a target-reliability weight.
* **A model that ignores the anchor still gets a mediocre score** by predicting one
  average pose per sequence — so aggregate RMSD alone cannot tell you the conditioning
  was learned. `anchor_sensitivity.py` is the check that can: it holds sequence and MHC
  fixed, changes only the anchor, and compares the model's response to AlphaFold's.

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
out     WIDEN 64 -> (single 384, pair 128) -> FROZEN multimer StructureModule
          -> backbone frames + sm["single"]
        sm["single"] -> TRAINABLE AngleResnet -> 7 torsions -> (SM's parameter-free
          geometry) -> full atom14   [bb-FAPE + sc-FAPE + supervised-chi]
        sm["single"] -> plddt_proj (identity-init adapter) -> FROZEN pLDDT head
          (exactly what AlphaFold feeds it)
```
Internal working width is **D=64** (single & pair), IPAs at **4 heads**; dims are only
widened to the frozen-module sizes (384/128) at the final projections. `forward ->
(ca, plddt_logits, None, frames, aux)` — the `DistillModel` contract plus a 5th `aux`
(angles / atom14 / sidechain frames). model_1's 4-tuple still works. PAE is `None`
(λ_pae=0), which drops a `[B,N,N,64]` zero tensor and an N²×64 cross-entropy per example.
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
| **angle_head** (torsions) | 166k | train | train | frozen |
| trunk (n_trunk=3) | 784k | train | train | frozen |
| **plddt_proj** (adapter, 384→384) | 148k | **frozen** | train | **train** |
| **sm** (StructureModule) | 2.02M | **frozen** | **frozen** | **frozen** |
| **plddt** (AF pLDDT head) | 73k | **frozen** | **frozen** | **frozen** |
| **trainable total** | | **1,100,752** | **1,248,592** | **147,840** |

`plddt_proj` is frozen in stage 1 because λ_plddt = 0 there: it would receive no gradient,
i.e. be a *silent unused parameter* (and force DDP `find_unused_parameters=True`).

`set_stage(stage)` in `model.py` applies this. (The opt-in `--unfreeze-sm-pct P
--unfreeze-sm-at E` can fine-tune the last P% of the SM from epoch E — tested at 10%/ep10,
it barely helped (0.52 vs 0.55 Å overfit), so it is OFF by default and the SM stays frozen.)

## Three-stage schedule
- **Stage 1 (structure ONLY)** — encoder + trunk + projections + `angle_head` trainable.
  **λ_plddt = 0 for the whole stage** and `plddt_proj` is frozen. Confidence-**filtered**
  subset; hasmig at full weight. Losses: bb-FAPE 0.5 + sc-FAPE 0.5 + chi 1.0.
- **Stage 2 (broader data + confidence)** — same trainable set **plus `plddt_proj`**;
  λ_plddt = 0.01; hasmig down-weighted 0.1. Structure losses unchanged.
- **Stage 3 (confidence-only)** — freeze everything except `plddt_proj`; the forward
  detaches the SM `single` so confidence learning cannot move the structure. All structure
  weights 0; λ_plddt = `--stage3-plddt-w` (1.0).

Each stage is a separate job; stage k+1 resumes stage k's checkpoint (loads weights,
fresh optimizer/scheduler because the stored `stage` differs).

### Stage 2 — two alternative experiments (each a separate 4-GPU job)

| | **Approach A** (`mm1_stage2_A`) | **Approach B** (`mm1_stage2_B`) |
|---|---|---|
| idea | keep it clean: train only on the good structures | use everything, but trust bad structures less |
| training data | confidence-**filtered** subset only | **FULL** dataset (no filter) |
| structural (FAPE) loss | uniform over the filtered set | per-structure **quality-weighted** by `w_n` |
| pLDDT CE | full weight on the filtered set | full weight on **all** structures |
| validation | filtered val subset | **all** val structures |
| flags | `--force-filter --filter-val` | `--struct-quality-weight` |
| run | `sbatch …/stage2_mm1.sbatch A` | `sbatch …/stage2_mm1.sbatch B` |

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

# 2b. side-chain targets for BOTH stores (required by --sidechains; ~6.3 GB total)
sbatch src/model/reprocess_sidechains.sbatch          # old store  -> teacher_atom14 + chi
# (hasmig's preprocess_hasmig.sbatch above already passes --sidechains)

# 3. stage 1 RETRAIN: leak-free + AlphaFold side chains (from scratch)
sbatch src/model_multimer_1/run_stage1_sidechain.sh
# legacy (leaky, no side chains):  sbatch src/model_multimer_1/train_mm1.sbatch 1

# 4. stage 2 — two experiments, each its own 4-GPU job (approach = first arg: A or B)
cp checkpoints_mm1/mm1_stage1/last.pt checkpoints_mm1/pretrained_stage1_last.pt
sbatch src/model_multimer_1/stage2_mm1.sbatch A
sbatch src/model_multimer_1/stage2_mm1.sbatch B

# 5. stage 3 (confidence-only), resuming the chosen stage-2 run:  args = STAGE RESUME
sbatch src/model_multimer_1/train_mm1.sbatch 3 checkpoints_mm1/mm1_stage2_A/last.pt
```

## Side chains (AlphaFold-style) — `--sidechains`
Previously the side chains were **never supervised**: `DistillLoss` had no chi/torsion term
and the H5 stored only `teacher_bb` (N,CA,C), so the frozen `sm.angle_resnet` emitted
torsions from an out-of-distribution `single` (chi1 was 10.8% correct vs a **22% random**
baseline — *worse than guessing*; all-atom RMSD 2.16 Å while Cα was 0.12 Å).

**Generation** (exactly AF2): a **trainable** `AngleResnet` predicts 7 torsions
(omega, phi, psi, chi1..4); atom14 is built with the SM's `torsion_angles_to_frames` +
`frames_and_literature_positions_to_atom14_pos`, which contain **no learned parameters** —
so the **StructureModule stays 100% frozen**. Bond lengths/angles are ideal constants, so
side-chain atom error *is* torsion error. Verified: rigid-group 0 (N,CA,C,CB) is bit-identical
to the SM's own atom14; groups 3..7 (O via psi, chi1..4) follow our torsions.

**Supervision** (AF2's own losses + weights): backbone FAPE 0.5 + `sidechain_loss` FAPE 0.5
(over the 8 rigid-group frames + atom14, with `compute_renamed_ground_truth` for symmetric
atoms, Alg. 26) + `supervised_chi_loss` 1.0 (chi_weight 0.5, angle_norm_weight 0.01, with the
`chi_pi_periodic` min-trick). AF enables the violation loss only in fine-tuning; so do we.

**Data**: preprocess with `--sidechains` -> `teacher_atom14` [N,14,3] fp16 (a *lossless*
repack of atom37; ≤0.008 Å quantization; 6.3 GB for 333k structures) + `teacher_chi`.
`sidechain_gt_from_atom14()` derives the rigid-group/alt/renaming tensors batched on-device.

**The pLDDT head had the same bug**: it read `plddt_proj(trunk_single)` where AF feeds it the
SM's own `single`. Rewired, with an identity-initialised adapter (stage 1 therefore reproduces
AF's confidence exactly). Peptide pLDDT-MAE fell 62–76 -> 8.0.

### Reading `eval_sidechains.py`
Report chi **split MHC vs peptide**, each next to its random baseline. The MHC is ~95% of the
residues, so a **pooled** chi number is near-perfect just by memorising one allele's rotamers.
Convergence test (48 train / 12 disjoint held-out, one allele — 150 ep):

| | chi1 (% within 40°) | random |
|---|---|---|
| untrained | 20.5% | 22.2% |
| MHC only (same allele as train) | 96.7% | 22.2% |
| **peptide only** (held-out peptides) | **80.9%** | 22.2% |

This proves the side-chain objective **trains**; it does **not** prove generalisation (those
peptides are single-residue point mutants of the training peptides). The real read is the
`two_axis` held-out fold after preprocessing the full store with `--sidechains`.

## Inference on data/test (full-atom PDB + RMSD)
`predict_test.py` runs a checkpoint on the 15 class-I examples in `data/test/`, writes a
**full-atom PDB** per example (atom14 → atom37; chain A = MHC, chain B = peptide;
b-factor = predicted pLDDT), Kabsch-superposes each prediction onto its reference PDB
**using the MHC Cα** (so the PDB overlays directly), and reports peptide / backbone /
all-atom RMSDs to `rmsd.csv`.
```bash
$PY src/model_multimer_1/predict_test.py --ckpt checkpoints_mm1/mm1_stage2_A/last.pt \
    --tag A --pep-frames identity --out-dir outputs/mm1_test
```
Always report the `--pep-frames identity` number — `teacher` is the leaky one (see the
warning at the top).

## Status: wiring validated end-to-end; TRAINING never completed (see the top of this file)
Local (pmgen2, 1 GPU, bf16) smoke + overfit: the full path (real hasmig zips → H5 → dataset
→ `InputEmbedderMultimer` → head-2 → trunk → frozen multimer SM → FAPE → backward → DDP) runs
clean. A clean 20-structure overfit at n_trunk 3 drove peptide Cα-RMSD to **~0.05 Å**,
confirming the Rigid3Array frames, `StructureModule(is_multimer=True)` I/O, and `*_mm.pt`
weight keys are wired correctly and that the frozen-SM design has ample capacity. The 2-GPU
`smoke_mm1.sbatch` passed on Raven (sharded DDP train + all-reduced val). Run the smokes
(step 2) before any full training.

**What this does NOT show.** Every number above is either a 20-structure overfit or a
smoke. The one full-scale leak-free attempt diverged and NaN'd before its first validation
pass (see CURRENT STATE at the top). "Ample capacity on a 20-structure overfit" is a
statement about memorisation, not generalisation — the held-out `two_axis` fold has never
been measured for this model.
