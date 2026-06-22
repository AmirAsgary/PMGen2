# Visualization

`plot_training.py` — compare training runs from their `metrics.csv` (+ `config.json`).

Each run dir (`checkpoints/<scheme>_fold<F>_variant<V>[_pw<W>]/`) holds a
`metrics.csv` written during training. This script reads many of them and draws
one figure: a grid of panels with **one row per metric**, **one column per
`--col` value** (default `peptide_weight`), and **one coloured line per `--hue`
value** (default `variant`). When several runs share a panel/colour (e.g. the 5
CV folds) they are averaged into a mean line + ±std band.

Both metrics.csv schemas are handled — older runs without the `pep_*` columns
just skip those metrics.

## Examples
```bash
PY=$(conda info --base)/envs/pmgen2/bin/python

# the 7 variants compared across the default loss curves (mean over folds),
# w=1 vs w=5 side by side
$PY src/visualization/plot_training.py --runs checkpoints -o training.png

# per-epoch eval metrics (Cα-RMSD, peptide RMSD, PAE-MAE, ...)
$PY src/visualization/plot_training.py --runs checkpoints --split val -o val.png

# one fold, raw per-step curves (no fold averaging)
$PY src/visualization/plot_training.py --runs checkpoints \
    --scheme two_axis --fold 1 --weight 1.0 --no-aggregate -o fold1.png

# focus on peptide metrics, columns = fold
$PY src/visualization/plot_training.py --runs checkpoints --split val --weight 5.0 \
    --metrics pep_ca_rmsd pep_pae_mae ca_rmsd --col fold -o peptide.png
```

## Key flags
- `--split {train,val}` — `train` = per-step loss curves; `val` = per-epoch eval metrics.
- `--hue` / `--col` — any of `variant`, `fold`, `scheme`, `peptide_weight`, `run`
  (`--col none` for a single column).
- `--metrics …` — override the default metric set.
- `--x {epoch,step,time}` — x axis (`epoch` = fractional epoch, comparable across folds).
- `--no-aggregate` / `--no-band` — draw every run / hide the ±std band.
- `--smooth 0.05` — moving-average for noisy train curves.
- Filters: `--scheme`, `--fold`, `--variant`, `--weight`.

The "best <hue>" callout in each panel marks the run/group with the best final
value (for metrics where lower is better).

### λ-normalised total
We cut `λ_plddt`/`λ_pae` from 0.1 → 0.01 mid-project, so the as-trained `total`
is not comparable across that change. `plot_training.py` therefore reconstructs a
`total_norm` column = the **raw** (un-weighted) `fape`/`plddt_ce`/`pae_ce` re-mixed
with one fixed reference λ (`REF_LAMBDAS = (1, 0.1, 0.1)`), and uses it by default.
The individual component columns (`fape`, `pep_ca_rmsd`, …) are already raw and so
are comparable as-is; only `total` was contaminated.

---

## `plot_overview.py` — model_2 curves + cross-model best comparison

`plot_training.py` only understands model_1's schema. `plot_overview.py` adds:
- **MHC-Diff (model_2) loss curves** (`model2_train.png`, `model2_val.png`) from the
  diffusion `metrics.csv` (`pep_coord`/`mhc_coord`/`torsion` + sampled RMSDs).
- A **best-model comparison** (`best_models_comparison.{png,csv}`) ranking every
  model_1 run and model_2 by peptide Cα-RMSD — the only metric the two models share
  (their losses are different objectives, so loss is NOT cross-comparable).

Two caveats are baked into the plot: (1) different CV folds = different val sets, so
absolute RMSD is only comparable **within** a fold (model_2 lives on fold 1, hatched
model_1 bars are also fold 1); (2) model_2's RMSD is measured with the **true MHC
given** (easier) — not 1:1 with model_1's own-MHC RMSD.

```bash
$PY src/visualization/plot_overview.py \
    --model1-runs tmp/checkpoints \
    --model2-csv  tmp/checkpoints_mhcdiff_06_21_2026/metrics.csv \
    --out-dir outputs/visualization
```

## Data-quality / stratified-eval helpers (`src/post_structure_prediction_processing/`)

- `check_data_quality.py` — peptide pLDDT + peptide↔MHC proximity over the H5 store,
  stratified by HLA cluster.
- `check_test_pdb_distance.py` — the same proximity metric on the `data/test` PDBs
  (AF predictions), overlaid on the H5-store distribution.
- `eval_stratified.py` — runs a trained model (`--model 1|2`) on val and stratifies
  peptide Cα-RMSD by teacher peptide pLDDT and peptide nearest-MHC distance, to show
  whether poor predictions track poor *data*.
