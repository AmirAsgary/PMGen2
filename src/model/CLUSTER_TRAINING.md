# Running PMGen-v2 distillation on the cluster

The "7 settings" = the **7 encoder variants** (one stripped pairformer block with
different pair-update ops; see `src/model/METHOD.md`). Each is a separate training
run so you can compare them. Everything trains **only the small encoder**; the AF2
structure module + pLDDT/PAE heads stay frozen (loaded from
`params/alphafold/*.pt`, which are in the repo).

All commands run from the **repo root**. Set:
```bash
PY=$(conda info --base)/envs/pmgen2/bin/python   # after installation.sh
```

## 0. One-time setup
```bash
bash installation.sh                       # creates the pmgen2 env
$PY src/model/encoder_test.py              # GPU smoke test (15 dummy examples)
```
`params/alphafold/{sm,plddt,pae}_af2.pt` ship in the repo, so no weight
extraction is needed. (Only if you want to rebuild them: get
`params_model_2_ptm.npz` and run `python src/afbuild/build.py`.)

## 1. One-time preprocessing → streamable HDF5 store
Turn the chunked PMGen teacher outputs into per-chunk HDF5 shards + `index.csv`.
Each chunk is independent → parallelise with a SLURM array, resumable.

```bash
# all chunks in one job:
$PY src/model/preprocess.py \
    --chunks-dir ~/projects/PMGen_2/data/pmgen_inputs/chunks \
    --out-dir   data/processed/h5_store
```

Parallel (one array task per chunk):
```bash
#!/bin/bash
#SBATCH --job-name=prep_%a
#SBATCH --array=1-222                 # number of chunks
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/prep.%A_%a.out
#SBATCH -e ./logs/prep.%A_%a.err
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2"
mamba activate pmgen2
$PY src/model/preprocess.py \
    --chunks-dir "$HOME/projects/PMGen_2/data/pmgen_inputs/chunks" \
    --out-dir    data/processed/h5_store \
    --chunks "chunk_${SLURM_ARRAY_TASK_ID}" \
    --no-merge                        # merge once at the end (next line)

# after the array finishes, merge the per-chunk *.index.csv into index.csv:
#   $PY src/model/preprocess.py --merge-only --out-dir data/processed/h5_store
```
(The shard for each `chunk_N` reads its outputs via `chunk_N/output/alphafold/<id>/`.
Missing/failed predictions are skipped and counted in the log.)

> Tip: the store is ~86 KB/example (~74 GB for 885k). For fastest training, stage
> it to node-local NVMe (`$SLURM_TMPDIR`) and point `--h5-dir` there.

### 1b. Recover failed predictions (optional)
A few chunks may have partly-failed PMGen jobs (the prep log shows
`X ok, Y missing, Z failed`). `missing` = the id's output dir doesn't exist;
`failed` = it exists but the `.pdb`/`.npy` are absent or don't match the
sequence. Build **balanced** retry inputs (failed ids spread evenly across
≤ `--max-jobs`, each shard kept in one chunk so outputs land in its dir):

```bash
python src/model/make_retry_inputs.py \
    --chunks-dir ~/projects/PMGen_2/data/pmgen_inputs/chunks \
    --h5-dir     data/processed/h5_store \
    --out-dir    data/processed/retry_inputs --max-jobs 32
# -> writes retry_*.tsv + manifest.tsv (+ incomplete_dirs.txt); prints --array=1-N
```
Then set `--array=1-N` in `src/model/retry_predict.sbatch` (it reads the manifest
per task and re-runs PMGen into each id's own chunk output dir) and submit it.
If PMGen skips ids whose folder already exists, remove the stale dirs first:
`xargs rm -rf < data/processed/retry_inputs/incomplete_dirs.txt`.
Afterwards, re-preprocess just those chunks (`--chunks chunk_117,... --overwrite
--no-merge`) and `--merge-only` again.

## 2. Train all 7 variants (one SLURM array)
```bash
#!/bin/bash
#SBATCH --job-name=pmgen2_distill
#SBATCH --array=1-7                   # variant = array index
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill_v%a.out
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/PMGen2"
$PY src/model/train.py \
    --h5-dir data/processed/h5_store \
    --scheme two_axis --fold 1 \
    --variant ${SLURM_ARRAY_TASK_ID} \
    --epochs 50 --bs 8 --lr 1e-3 --grad-clip 1.0 \
    --num-workers 8 --amp
```
`--array=1-7` launches the 7 variants in parallel, one GPU each.

### Choosing the data split
- `--scheme {two_axis,hla_only}` and `--fold {1..5}` pick which cluster-aware
  split to train on (val = that fold; test held out). All anchor combinations of
  a (peptide, MHC) pair always land in the same split.
- To sweep variants × folds, use a 2-D array, e.g. `--array=0-34` and decode
  `variant=$((idx/5+1))`, `fold=$((idx%5+1))`.

### Single run (no SLURM)
```bash
$PY src/model/train.py --h5-dir data/processed/h5_store \
    --scheme two_axis --fold 1 --variant 7 --epochs 50 --bs 8 --num-workers 8 --amp
```

### Local sanity without the H5 store
```bash
$PY src/model/train.py --dummy --variant 7 --epochs 20 --bs 3   # 15 dummy examples
```

## Useful flags (`train.py --help`)
`--variant 1..7`, `--scheme`, `--fold`, `--h5-dir` | `--dummy` | `--af-root`,
`--epochs --bs --lr --lambdas FAPE PLDDT PAE --weight-decay --grad-clip --amp
--num-workers --seed --device`.
