# PMGen2

Peptide–MHC dataset curation + **PMGen v2**: a trunk-distillation model that
replaces AlphaFold2's Evoformer with a small trainable encoder while **reusing
AF2's structure module + pLDDT/PAE heads frozen**. We train only the encoder to
reproduce PMGen (AF2-based) teacher structures from sequence + anchors.

## Install (cluster)

```bash
bash installation.sh            # creates the `pmgen2` conda env (CUDA torch + deps)
# CUDA tag / torch version are configurable: bash installation.sh pmgen2 cu124
```
OpenFold lives in [`openfold/`](openfold/README.md) as a **vendored, modified**
copy used from `PYTHONPATH` (no pip install, no CUDA-kernel build). The frozen AF2
weights (`params/alphafold/{sm,plddt,pae}_af2.pt`, ~8 MB) ship in the repo.

Smoke test (needs a GPU):
```bash
PY=$(conda info --base)/envs/pmgen2/bin/python
$PY src/model/encoder_test.py
$PY src/model/train.py --dummy --variant 7 --epochs 5 --bs 3
```

## Repository layout

| Path | What |
|------|------|
| `src/model/` | PMGen v2: encoder (7 variants), frozen wrapper, 3-term loss (FAPE + pLDDT-CE + PAE-CE), dataset + **HDF5 preprocessing/store**, train loop. Entry points `train.py`, `preprocess.py`; gates `*_test.py`. See `src/model/METHOD.md`, `src/model/CLUSTER_TRAINING.md`. |
| `src/pdb/` | Teacher-PDB parser → tensors + sanity/overfit scripts. See `src/pdb/METHOD.md`. |
| `src/afbuild/` | `load_frozen_fold` (frozen AF2 SM + heads) + one-time weight extraction. |
| `src/data_processing/` | pMHC class-I dataset assembly + cluster-aware splits (see `METHODS.md`). |
| `openfold/` | Vendored, modified OpenFold (Apache-2.0). |
| `params/alphafold/` | Frozen AF2 weights (`*.pt` tracked; `*.npz` is git-ignored — rebuild via `src/afbuild/build.py`). |
| `data/processed/` | Cluster-aware split definitions (`two_axis/`, `hla_only/`, `base_ids_ordered.csv`). Big tables are git-ignored. |
| `data/test/` | 15 dummy class-I examples for smoke tests / `--dummy`. |

## Train (the 7 encoder variants)

The "7 settings" are the 7 encoder variants (different pair-update ops). Workflow:

1. **Preprocess once** → streamable HDF5 store (one shard per PMGen chunk, keyed by
   anchor id; splits are id-lists, so data is stored once):
   ```bash
   $PY src/model/preprocess.py --chunks-dir <chunks> --out-dir data/processed/h5_store
   ```
2. **Train a variant** (encoder-only; AF2 stack frozen):
   ```bash
   $PY src/model/train.py --h5-dir data/processed/h5_store \
       --scheme two_axis --fold 1 --variant 7 --epochs 50 --bs 8 --num-workers 8 --amp
   ```

Full cluster recipe (SLURM arrays over chunks / variants): **[`src/model/CLUSTER_TRAINING.md`](src/model/CLUSTER_TRAINING.md)**.

## Notes
- Large data (raw parquet/clusters, full tables, the HDF5 store, `params_*.npz`,
  the anchor TSVs) are **git-ignored** — they're regenerated or live on the
  cluster. Only code + small split/weight artifacts are pushed.
- Methods write-up: [`METHODS.md`](METHODS.md) (data) and per-module `METHOD.md`.
