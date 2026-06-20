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
