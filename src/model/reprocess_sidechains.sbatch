#!/bin/bash -l
# ============================================================================
# Re-preprocess the PMGen teacher chunks INTO A NEW store that also carries
# side-chain torsions (teacher_chi [N,4,2] + teacher_chi_mask [N,4]) for model_2.
# Identical layout to the backbone store, one shard per chunk, run in parallel.
#
# Writes to data/processed/h5_store_sc (kept separate from the backbone-only
# h5_store so model_1 is untouched). After the array finishes, merge once:
#   PY=$(conda info --base)/envs/pmgen2/bin/python
#   $PY src/model/preprocess.py --merge-only --out-dir data/processed/h5_store_sc
#
# Submit:  sbatch src/model/reprocess_sidechains.sbatch   (set --array to #chunks)
# ============================================================================
#SBATCH --job-name=pmgen2_reproc_sc
#SBATCH --array=1-222%32           # one task per chunk_N; tune to #chunks / quota
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:50:00
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

# one chunk -> its shard; --sidechains stores teacher_chi; merge later (--no-merge)
$PY src/model/preprocess.py \
    --chunks-dir "$CHUNKS_DIR" \
    --out-dir    "$OUT" \
    --chunks     "$CHUNK" \
    --sidechains \
    --no-merge

exit $?
