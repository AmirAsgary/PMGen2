# model_multimer_1

> ## CURRENT STATE (2026-08-23) — stage 2 complete, confidence head working
> **Best checkpoint: `checkpoints_mm1/ARCHIVE/stage2b_best.pt`** (stage 2, gstep 96,096,
> epoch 5/5, LR annealed to exactly 0). All numbers on the same seeded 1,500-structure
> `two_axis` val subset, scored by `eval_checkpoints.py` in one job so they are comparable.
>
> | checkpoint | gstep | val total | val pep-RMSD | val pLDDT-MAE | vm total | vm pep-RMSD | vm pLDDT-MAE |
> |---|---|---|---|---|---|---|---|
> | `ARCHIVE/stage1_best_20260821.pt` | 60,816 | 0.300 | 4.76 Å | 18.64 | 0.158 | 1.34 Å | 12.98 |
> | `ARCHIVE/stage2_best.pt` (λ_plddt=0.01, 5:1) | 92,546 | 0.232 | 3.65 Å | 6.75 | 0.148 | 1.26 Å | 7.99 |
> | `ARCHIVE/stage2b_best.pt` (λ_plddt=0.01, **50:1**) | 96,096 | **0.222** | **3.47 Å** | **5.80** | **0.143** | **1.21 Å** | **6.53** |
>
> Adding the pLDDT CE at λ=0.01 did **not** cost structural accuracy — it improved it
> (4.76 → 3.47 Å full-fold, 1.34 → 1.21 Å on the matched subset) while cutting pLDDT MAE
> by 3.2×. Weighting peptide 50× MHC inside the CE (`--plddt-peptide-ratio 50`, peptide
> 19.9% → 71.3% of the CE) gave a further improvement on every column.
>
> ### On the 15 `data/test` structures (`outputs/mm1_test/final_identity/`)
> | metric | stage 1 | **stage 2b** |
> |---|---|---|
> | peptide Cα RMSD | 1.851 Å mean / 0.962 Å median | **1.416 / 0.882** |
> | peptide backbone RMSD | — | 1.362 / 0.867 |
> | peptide all-atom RMSD | 2.674 Å mean / 1.985 Å median | **2.449 / 1.896** |
> | MHC Cα RMSD | 0.120 Å | 0.124 Å |
> | whole-complex all-atom | — | 0.792 Å mean |
> | leak Δ (teacher_bb +10 Å) | 0.000 Å | **0.000 Å** |
> | latency, batch 1 | 141.6 ms | 126.4 ms model + 12.1 ms PDB write |
>
> 9/15 peptides under 1 Å Cα. The two worst are the `3;9` and `2;8` anchor variants
> (4.81 Å, 2.53 Å) — extreme anchors the model is least confident about, and it says so:
> their predicted pLDDT is the lowest in the set (66.0, 58.0).
>
> ### Peptide confidence (this is the number that matters — MHC is ~95% of residues)
> Measured with `plddt_corr.py` on 400 val structures, peptide residues only:
>
> | | stage 1 | stage 2 (5:1) | **stage 2b (50:1)** |
> |---|---|---|---|
> | per-residue vs teacher pLDDT (Pearson) | +0.272 | +0.774 | **+0.845** |
> | per-structure vs teacher pLDDT (Pearson) | +0.401 | +0.794 | **+0.865** |
> | **vs ACTUAL peptide Cα-RMSD (Spearman)** | −0.448 | −0.609 | **−0.635** |
> | MAE vs teacher | 15.07 | 5.76 | **4.54** |
>
> The third row is the practical one: it is what lets a user throw away a bad prediction
> without knowing the answer. On the FILTERED (high-confidence) slice the ranking
> correlation is −0.578 with a 46-point pLDDT spread, i.e. it still discriminates inside
> the easy regime rather than saturating.
>
> ### Anchor conditioning — now learned (`anchor_sensitivity.py`, 120 sibling pairs)
> | | untrained control | **stage 2b** |
> |---|---|---|
> | model response to an anchor change | 12.24 Å (chaos) | 2.41 Å |
> | AlphaFold's own response | 3.54 Å | 3.54 Å |
> | ratio model/data | 3.46× | 0.68× |
> | **accuracy gain from the RIGHT anchor** | −0.32 Å (none) | **+1.11 Å** |
> | right anchor wins on | 49% (chance) | **88%** of complexes |
>
> Read the *gain* row, never the ratio: a random model scores a high ratio because it is
> chaotic. Stage 2b under-responds in magnitude (0.68×) but uses the anchor correctly.
>
> > ### The instability that killed eight runs: FIXED
> The trunk's self-attention received **raw, unnormalised input** — `nn.MultiheadAttention`
> does no internal normalisation and none had been added, so attention logits grew as
> `||s||²/√d` while the residual stream grew unchecked (measured across the 3 blocks:
> 107→335 in a stable checkpoint, 311→1463 in a pre-collapse one). The IPA branch was
> already guarded by `s_norm`; only attention was exposed. AlphaFold normalises before
> every attention. Fixed as `attn_norm` (default ON, `--no-attn-norm` to disable).
>
> **Ruled out by measurement, not argument** (all of these were tried and failed):
> gradient spikes (the *stable* checkpoint spiked MORE: 31 vs 18 windows >10× median);
> pathological data (ρ=+0.02 between checkpoints, 0/20 top spikers shared; the structures
> fed during an explosion were statistically identical to healthy ones); the `1/||s||`
> angle-norm runaway (ρ=−0.19); sharp minima (loss moves <0.006 under a 3% weight
> perturbation); optimizer state (a continued-optimizer arm died *sooner*); data order;
> learning rate (3e-4 died at 4,500 steps, 1e-4 at 11,500 — a 2.6× delay for a 3× LR cut,
> i.e. a fixed `lr×steps` budget, not a cure); and bounding the trunk *output* scale
> (died at 3,300, EARLIER than the control).
>
> ### Also fixed
> * **Side-chain losses escaped every weighting.** `sc_fape` and `chi` were plain `.mean()`s
>   over all residues of all examples, bypassing `--peptide-weight`, `--hasmig-weight` and
>   `--struct-quality-weight` — together ~71% of the total loss and ~95% MHC. Approach B's
>   entire premise was inert. Fixed via `_sidechain_fape_per_example` (routes the weight
>   through `compute_fape`'s `pair_mask`, which enters the normaliser) and
>   `_supervised_chi_loss_per_example` (openfold reduces the batch to a SCALAR internally,
>   so no per-example weight could reach it at all).
> * **`evaluate()` ran without `torch.no_grad()`** — every val batch built and retained the
>   full autograd graph through the frozen AF stack.
> * **The pLDDT adapter is gone.** AlphaFold feeds the frozen head the SM's `single`
>   DIRECTLY (openfold `heads.py:57`), and `out["single"]` is already refined by all 8 SM
>   blocks. The identity-initialised adapter was verified a bit-exact no-op at init, so it
>   could only ever drift the input off the frozen head's training distribution.
>   `--plddt-adapter` restores it. **Stage 3 trained ONLY that adapter and now refuses to
>   run without it** rather than silently training nothing.
>
> ### Schedules must ANNEAL — the single strongest predictor of survival
> | run | peak lr | final lr | decayed | fate |
> |---|---|---|---|---|
> | `mm1_s1_full` | 5.0e-4 | 4.87e-4 | 2.6% | died |
> | `mm1_stage1_sc` (July) | 5.0e-4 | 4.96e-4 | 0.8% | died |
> | `trig_A/B/C` | 3.0e-4 | 3.00e-4 | 0.0% | died |
> | `mm1_s1_short` | 5.0e-4 | **0.00** | **100%** | **survived** |
>
> Every long run before 2026-08-20 set `T_max` for 6 epochs against a 24 h wall clock and
> truncated at ~15% — constant-LR runs in disguise. **Always size `--epochs` to what the
> job will actually execute.**


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
        sm["single"] -> FROZEN pLDDT head, DIRECTLY (no adapter) — exactly what
          AlphaFold does: heads.py:57 `self.plddt(outputs["sm"]["single"])`
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

| module | params | Stage 1 | Stage 2 |
|---|--:|:--:|:--:|
| embedder (head-1) | 33.5k | train | train |
| head-2 (MHC encoder) | 56.3k | train | train |
| s_proj / z_proj | 27.2k | train | train |
| sm_s / sm_z (widen→SM) | 33.3k | train | train |
| **angle_head** (torsions) | 166k | train | train |
| trunk (n_trunk=3, incl. 3x pre-attn LayerNorm) | 784k | train | train |
| **sm** (StructureModule) | 2.02M | **FROZEN** | **FROZEN** |
| **plddt** (AF pLDDT head) | 73k | **FROZEN** | **FROZEN** |
| **trainable total** | | **1,101,136** | **1,101,136** |

Stages 1 and 2 now train the SAME parameters — the only difference is the LOSS
(lambda_plddt 0 vs 0.01) and the data weighting. The old stage-2 "plus plddt_proj" no
longer exists: the adapter was removed, so there is nothing extra to unfreeze. Stage 3
is gone for the same reason (it trained only that adapter).


`set_stage(stage)` in `model.py` applies this. (The opt-in `--unfreeze-sm-pct P
--unfreeze-sm-at E` can fine-tune the last P% of the SM from epoch E — tested at 10%/ep10,
it barely helped (0.52 vs 0.55 Å overfit), so it is OFF by default and the SM stays frozen.)

## Are the results real? — adversarial verification (2026-08-21)

A high predicted-vs-true pLDDT correlation is exactly what a leak looks like, so it was
checked adversarially rather than trusted. `verify_results.py` runs all three:

**1. No leakage.** Every `teacher_*` field randomised on REAL val structures; the pLDDT
LOGITS *and* atom14 must be bit-identical:

| randomised field | max abs delta (pLDDT logits, atom14) |
|---|---|
| teacher_plddt / atom14 / atom14_mask / chi / chi_mask / ca / pae | **0.000e+00** |

> **A GAP THIS FOUND.** `_leak_check` historically asserted only on `atom14`. A leak into
> the CONFIDENCE head would have passed unnoticed while making the pLDDT correlation
> trivially fabricated. It now asserts on `plddt_logits` for every teacher field too.

**2. No memorisation.** train∩val: **0** shared `base_id`, **0** shared exact ids
(204,874 train ids / 189,733 base_ids vs 30,239 val ids / 9,160 base_ids).

**3. Not a metric artifact.** The same measurement on random weights:

| | per-residue r | rho | per-structure r |
|---|---|---|---|
| trained | **+0.750** | +0.726 | +0.720 |
| **untrained control** | **−0.023** | −0.026 | −0.079 |

### What the val numbers DO and DO NOT establish
`splits_metadata.json`: `mode=two_axis`, `split_axis=hla+peptide`, but the **CV val fold
holds out the PEPTIDE axis only** (`cv_val_peptide_frac: 0.2`). The HLA axis is held out
in **`test.csv`** (`test_hla_frac: 0.1`), which training never touches. Measured: **363 of
871 val alleles also appear in train** (with different peptides) — by design, not a defect.

| number | what it actually measures |
|---|---|
| **val / val-matched** (e.g. 1.34 Å) | unseen **PEPTIDES**, largely on **SEEN alleles** |
| **`data/test`** (the 15 PDBs, 1.851 Å mean / 0.962 Å median) | both axes held out — the stricter, more conservative figure |

Quote `data/test` externally. Quote val-matched for peptide generalisation, and say so.

## Stage schedule (REVISED 2026-08-21 — supersedes the original three-stage plan)

**Stage 1 — structure only.** `--stage 1`, lambda_plddt = 0. Confidence-**filtered**
subset (`burial >= 0.65 AND pep-pLDDT > 70`, ~8.8% of the store), hasmig at full weight.
Losses: bb-FAPE 0.5 + sc-FAPE 0.5 + chi 1.0.
Run: `run_stage1_long24.sh` (bs 12, 4 GPUs, annealed).

**Stage 2 — add confidence, keep the structure objective.** `--stage 2`, and exactly one
thing changes: lambda_plddt 0 -> **0.01**. Run: `run_stage2.sh`.

    --plddt-w 0.01           MEASURED, not inherited. On real batches the pLDDT CE
                             gradient is 15.46x the structural one at lambda=1, so 0.01
                             puts confidence at ~15.5% of the structural gradient.
    --hasmig-weight 1.0      stage 2 DEFAULTS to 0.1; we keep 1.0. hasmig is 130,993
                             structures at median pep-pLDDT 83.4 with ONE structure per
                             complex — the cleanest data here. The 0.1 down-weight was
                             never justified by a measurement.
    NO --force-filter        DELIBERATE. Stage 1 trains only on high-confidence
                             structures; a model that never sees the low-confidence tail
                             cannot learn to REPORT low confidence. Non-discriminative
                             pLDDT is exactly the open gap (test set: predicted 72.8-79.6
                             while true error spans 0.5-5.1 A).
    --struct-quality-weight  with the filter gone, this scales the STRUCTURAL loss per
                             structure by w_n, leaving the pLDDT CE at FULL UNIFORM weight
                             on every structure. Structure objective stays effectively as
                             in stage 1 (bad structures contribute ~w_min=0.05); the
                             confidence term sees the whole range at a flat 0.01,
                             independent of quality and burial.

**Stage 3 — REMOVED.** It trained *only* `plddt_proj`, the adapter in front of the frozen
pLDDT head. That adapter is gone (AlphaFold feeds the head the SM's `single` directly),
so stage 3 has zero trainable parameters and `set_stage(3)` now **raises** rather than
silently training nothing. `--plddt-adapter` restores the old path if you want it back.

### Why confidence learning is indirect here — and how much reaches the target
The frozen head reads `out["single"]`, which the SM builds over 8 blocks as
`s = s + ipa(s, z, ...)` with the pair rep entering as the IPA's attention bias. Measured
on real batches, where each loss deposits its gradient:

| loss | single path | pair path | other | single share |
|---|---|---|---|---|
| structure | 0.694 | 0.921 | 0.407 | 34.3% |
| **pLDDT CE** | **1.402** | **19.414** | 2.068 | **6.1%** |

So **94% of the confidence gradient lands outside the single path**. At lambda 0.01 the
single representation sees roughly 1% of the structural signal. Raising lambda to
compensate would put the *total* confidence gradient above the structural one, mostly as
pair updates.

Encouraging counter-evidence: during stage 1, with lambda_plddt = 0 and no confidence
supervision at all, `pep-pLDDT-MAE` on the held-out fold fell **25.08 -> 18.72** purely
because the SM's `single` improved. Confidence may improve largely as a by-product of
representation quality — measure whether the pLDDT correlation actually moves in stage 2
before reaching for a bigger lambda.

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

# 5. stage 2 (confidence). Stage 3 is REMOVED — see the stage schedule.
sbatch src/model_multimer_1/run_stage2.sh
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

**The pLDDT head had the same bug**: it read `plddt_proj(trunk_single)` where AF feeds it
the SM's own `single`. Rewired to read `sm["single"]`. The identity-initialised adapter
that briefly sat in between has since been REMOVED (see the stage schedule): AF feeds the
frozen head the SM's single directly, and the adapter was verified a bit-exact no-op at
init, so it could only ever drift the input off the head's training distribution.
Peptide pLDDT-MAE fell 62-76 -> 8.0 when this was first rewired.

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

## Status (2026-08-23)

Stages 1 and 2 are **complete and validated**. Stage 1: 15 annealed epochs at bs=12 over
the 204,874 filtered train set. Stage 2: 5 annealed epochs at bs=12 over the full
967,149-structure unfiltered set (the burial/pLDDT filter is dropped in stage 2 and
survives only as the `--struct-quality-weight` soft weight on FAPE), with the pLDDT CE at
lambda=0.01 and peptide weighted 50x MHC inside the CE. Neither run overfitted, neither
exploded, and both annealed the LR to exactly 0.

**What stage 2 bought** (see the table at the top): peptide pLDDT MAE 15.07 -> 4.54,
per-structure correlation with the teacher +0.401 -> +0.865, discrimination against ACTUAL
error rho -0.448 -> -0.635 — and structural accuracy IMPROVED at the same time
(val pep-RMSD 4.76 -> 3.47 A, test-set peptide Ca 1.851 -> 1.416 A). The concern that a
confidence term would trade away geometry did not materialise at lambda=0.01.

**Open items**, in rough priority:
1. **Batch-1 latency misses the 10-50 ms target**: 126 ms of model + 12 ms of PDB writing,
   of which ~74% is the FROZEN AlphaFold decoder and not reducible by training. Batching
   fixes it (37 ms at bs=4, 24 ms at bs=8), so screening workloads are fine; interactive
   single-structure prediction is not. The only real lever is fewer SM blocks, which means
   fine-tuning the decoder and losing the "frozen AF" generalizability guarantee.
2. **The anchor conditioning under-responds in magnitude**: 0.68x AlphaFold's displacement
   for the same anchor change. Direction and selection are correct (+1.11 A gain, 88% of
   complexes), but the hardest test cases are consistently the unusual anchors (`3;9`,
   `2;8`), which is exactly where under-response costs the most — the two worst test
   peptides are both of these.
3. **The tail is heavy.** Test-set peptide Ca is 0.88 A at the median but 1.42 A at the
   mean; val is 3.47 A because the unfiltered fold contains genuinely unpredictable
   AlphaFold outputs (true RMSD up to 42 A). The model now flags these correctly rather
   than hiding them, but it does not solve them. The incoming 400k high-quality diverse
   dataset is the intended fix.

### Diagnostics added (all in `src/model_multimer_1/`)
| script | answers |
|---|---|
| `anchor_sensitivity.py` | is the anchor conditioning LEARNED? (holds sequence+MHC fixed, changes only the anchor; includes an untrained control) |
| `mhc_channel_check.py` | does peptide pose leak in through the CO-FOLDED MHC input, and what is the deployment penalty with a foreign groove? |
| `locate_explosion.py` | per-stage activation and grad-wrt-activation norms — WHERE a gradient grows |
| `diagnose_collapse.py` | SM-input drift + loss-landscape sharpness across checkpoints |
| `measure_plddt_balance.py` | what lambda_plddt should be, and whether the CE trains the single or pair path |
| `bench_batch.py` | is bs>1 CORRECT (batched loss == mean of individual), and how fast |
| `bench_latency.py` | ms/structure vs the 10-50 ms target, split trainable vs frozen |
| `eval_checkpoints.py` | scores checkpoints on ONE common val set, auto-detecting each one's architecture from its keys |

**Architecture must match the checkpoint when evaluating.** `strict=False` silently leaves
mismatched modules at init, so scoring a pre-fix checkpoint under the new default
`attn_norm=True` understates it (measured: 1.93 -> 2.03 A). `eval_checkpoints.py` detects
`attn_norm` / `plddt_proj` from the state dict; `--angle-input` is not a parameter and
still has to be passed.
