#!/bin/bash -l
# ============================================================================
# Re-preprocess the PMGen teacher chunks INTO A NEW store that also carries the
# AF2 side-chain targets, for model_multimer_1 --sidechains training:
#   teacher_atom14 [N,14,3] fp16 + mask   (side-chain FAPE + symmetric renaming)
#   teacher_chi    [N,4,2]  fp16 + mask    (supervised chi)
# Identical layout to the backbone store, one shard per chunk, run in parallel.
#
# Writes to data/processed/h5_store_sc (kept SEPARATE from the backbone-only
# h5_store so model_1 is untouched). This is the OLD (two_axis) store that
# provides VALIDATION for model_multimer_1 -> point train_mm1.sbatch's H5 at it.
#
# After the array finishes, MERGE the per-chunk indices into one index.csv (the
# training code needs the merged index; per-chunk *.index.csv alone won't be found):
#   PY=$(conda info --base)/envs/pmgen2/bin/python
#   $PY src/model/preprocess.py --merge-only --out-dir data/processed/h5_store_sc
#
# Set --array to your ACTUAL number of chunks (ls "$CHUNKS_DIR"/chunk_* | wc -l).
# Submit:  sbatch src/model/reprocess_sidechains.sbatch
# ============================================================================
#SBATCH --job-name=pmgen2_reproc_sc
#SBATCH --array=1-222%16           # one task per chunk_N; tune to #chunks / quota
#SBATCH --nodes=1
# Request a GPU node like the training jobs: on Raven a CPU-only job lands in a
# small/shared QOS with a group-CPU cap (-> pending 'QOSGrpCpuLimit'). The GPU is
# left IDLE here (the parse is CPU work) but it routes the job to your allocation.
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00           # side-chain χ extraction is ~2x the backbone parse
#SBATCH --output=logs/reproc/%A_%a.out
#SBATCH --error=logs/reproc/%A_%a.err

CHUNKS_DIR="$HOME/projects/PMGen_2/data/pmgen_inputs/chunks"
OUT="data/processed/h5_store_sc"

mkdir -p logs/reproc "$OUT"
export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

CHUNK="chunk_${SLURM_ARRAY_TASK_ID}"
echo "=== $(date '+%F %T') | reprocessing ${CHUNK} (with side chains) -> ${OUT} ==="

# one chunk -> its shard; --sidechains stores teacher_atom14 + teacher_chi; merge later
$PY src/model/preprocess.py \
    --chunks-dir "$CHUNKS_DIR" \
    --out-dir    "$OUT" \
    --chunks     "$CHUNK" \
    --sidechains \
    --no-merge

exit $?
