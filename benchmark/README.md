# pRMSD benchmark — model_multimer_1 vs published methods

Run date 2026-08-24 · checkpoint `checkpoints_mm1/ARCHIVE/stage2b_best.pt` (gstep 96,096, stage 2)

## Headline

176 pMHC-I complexes, every method scored on the identical set, peptide RMSD measured
after superposing **on the MHC only**.

| method | Cα mean | Cα median | all-atom mean | all-atom median | Cα < 1 Å | Cα < 2 Å |
|---|---|---|---|---|---|---|
| PMGen+pLDDT | 1.038 | **0.620** | 1.898 | **1.378** | 67.6% | 85.2% |
| PMGen | **0.983** | 0.647 | **1.836** | 1.386 | 68.2% | 85.2% |
| Tfold | 1.275 | 0.732 | 2.158 | 1.635 | 61.4% | 88.6% |
| MHC-Fine | 1.022 | 0.757 | 1.954 | 1.656 | 64.8% | **89.8%** |
| **PMGen2-distilled (this work)** | **1.203** | **1.008** | **2.133** | **1.851** | **49.4%** | **87.5%** |
| PMGen2-distilled, foreign MHC groove | 1.260 | 1.021 | 2.165 | 1.857 | 48.9% | 85.8% |
| PMGen+TE | 1.209 | 1.047 | 2.195 | 1.847 | 47.7% | 88.1% |
| Pandora | 1.643 | 1.614 | 3.044 | 3.146 | 22.2% | 72.2% |
| AF-Multimer | 3.512 | 2.286 | 4.655 | 3.936 | 18.8% | 40.9% |

**5th of 8 on median Cα, 4th on mean Cα and mean all-atom, at 28.7 ms/structure** —
inside the 10–50 ms design target and roughly two orders of magnitude below the
AlphaFold-based methods it is distilled from. It beats PMGen+TE, Pandora and
AF-Multimer outright and trails PMGen / MHC-Fine / Tfold.

**Where it loses is the easy cases, not the hard ones.** Its Q3 (1.473 Å) is
indistinguishable from PMGen's (1.472 Å) and better than Tfold's tail; its Q1
(0.652 Å) is well behind PMGen's (0.389 Å). The distillation reproduces the
teacher's failure modes but not its sub-Ångström precision — it does not turn a
0.4 Å answer into a 3 Å one, it turns it into a 0.9 Å one.

## Protocol

1. **References.** `prepare_refs.py` rewrites each `reference_pdbs/{PDB}.pdb` (a
   single crystal chain C, MHC then peptide) into the PMGen teacher convention —
   chain A, MHC numbered 1..n_mhc, peptide restarting at n_mhc+201 so the ≥150 jump
   marks the chain break; altloc collapsed to the first conformer. The **same file**
   is then read both by `parse_example` (model input) and by openfold's
   `from_pdb_string` (RMSD reference), so the residue orderings cannot drift apart.
   All 176 sequences matched the TSV exactly, no insertion codes, no missing
   backbone atoms.
2. **Anchor enumeration.** Each complex is predicted once per anchor hypothesis:
   `(1,L) (1,L-1) (1,L-2) (2,L) (2,L-1) (3,L)` for L ≥ 9, and `(1,L) (1,L-1) (2,L)`
   for the 9 octamers. The variants of one complex are run as a single batch.
3. **Selection.** The variant with the highest **mean peptide pLDDT** is kept.
   (Peptide, not complex — the MHC is ~95% of residues and its pLDDT is nearly
   constant, so complex pLDDT barely discriminates. Complex pLDDT is recorded in
   `results_*_per_anchor.csv` for anyone who wants the other convention.)
4. **Scoring.** Kabsch superposition on **MHC Cα only**, then peptide Cα RMSD and
   peptide all-atom RMSD over heavy atoms present in both prediction and crystal.
   No side-chain symmetry correction is applied.

## The MHC-input question, and why the answer is reassuring

This model takes the MHC backbone as an **input** — that is the PMGen design, not a
shortcut — so the benchmark had to establish that it is not quietly reading the
answer out of a groove that has already relaxed around the true peptide.

Two runs, identical in every other respect:

| MHC input | Cα mean | Cα median | all-atom mean |
|---|---|---|---|
| the target's own crystal groove | 1.203 | 1.008 | 2.133 |
| **a foreign groove** from the most MHC-similar *other* benchmark complex, carrying a different peptide | 1.260 | 1.021 | 2.165 |

Handing the model a groove from a different complex costs **0.06 Å**. The peptide
pose is not being read off the MHC input. The peptide's own backbone frames are
withheld throughout (`--pep-frames identity`), and the leak test — teacher frames
displaced +10 Å and every target randomised — returned **0.000e+00 Å** on all 176.

## Training-set contamination

48 of the 176 benchmark peptide sequences appear in the `h5_store_hasmig` training
store (all hasmig ids go to train, unsplit). Zero appear in `h5_store_sc`. Note the
training targets were AlphaFold predictions, never these crystal structures.

The 48 are an easier subset **for every method**, which is what rules the effect out:

| subset | ours Cα median | PMGen Cα median |
|---|---|---|
| peptide never seen (n=128) | 1.034 | 0.665 |
| peptide seen in hasmig (n=48) | 0.944 | 0.608 |

Our gain from the seen subset (0.09 Å) is the same order as PMGen's (0.06 Å), and the
method ranking is unchanged in both halves. Per-complex flags: `contamination_flags.csv`.

## Files

| file | contents |
|---|---|
| `benchmark_boxplots.png` / `.pdf` | the two-panel figure (Cα and all-atom) |
| `benchmark_summary.csv` | per-method mean/median/Q1/Q3/fraction-under-threshold |
| `benchmark_all_methods_long.csv` | tidy: one row per (complex, method) |
| `results_mm1_crystal.csv` | our per-complex results, own-groove input |
| `results_mm1_template.csv` | our per-complex results, foreign-groove input |
| `results_mm1_*_per_anchor.csv` | every anchor hypothesis, before selection |
| `contamination_flags.csv` | per-complex `pep_seen_in_train` |
| `predictions/mm1_crystal/*.pdb` | full-atom predictions, MHC-superposed, b-factor = predicted pLDDT |
| `converted_pdbs/` | references in the PMGen convention (regenerate with `prepare_refs.py`) |

## Reproducing

```bash
$PY benchmark/prepare_refs.py            # once
sbatch benchmark/run_benchmark.sbatch    # both MHC-input conditions, ~4 min on 1 A100
$PY benchmark/make_figures.py
```

## Not in git

`reference_pdbs/` (926 public crystal structures, 243 MB), `converted_pdbs/` and
`predictions/` are gitignored. The first is external input; the other two are
regenerated by `prepare_refs.py` and `run_benchmark.sbatch`. Everything needed to
reproduce the numbers — the TSV, the three scripts and the result CSVs — is tracked.
