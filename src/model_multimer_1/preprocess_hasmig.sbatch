#!/bin/bash -l
# ============================================================================
# Parse the hasmig_mhcs dataset (one zip per id) into HDF5 shards, in parallel.
# Array task c writes hasmig_<c>.h5 + hasmig_<c>.index.csv over rows c::N_CHUNKS.
# After the array finishes, merge the per-chunk indices:
#     python src/model_multimer_1/preprocess_hasmig.py --merge \
#         --out-dir data/processed/h5_store_hasmig \
#         --csv x --zip-dir x            # (csv/zip-dir ignored for --merge)
#
#   FIRST: extract the tarball once (it holds the per-id zips):
#     tar -xzf data/hasmig_mhcs/burial_score_output.tar.gz -C data/hasmig_mhcs/
#   (the dir data/hasmig_mhcs/burial_score_output/ already holds the .zip files)
#
# Submit:  sbatch src/model_multimer_1/preprocess_hasmig.sbatch
# ============================================================================
#SBATCH --job-name=hasmig_h5
#SBATCH --array=0-63%16             # 64 chunks; 16 concurrent (tune to quota)
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1           # CPU work, but route through gpu allocation
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hasmig/%A_%a.out
#SBATCH --error=logs/hasmig/%A_%a.err

N_CHUNKS=64
CSV=${CSV:-data/hasmig_mhcs/filtered_burial_scores.csv}
ZIPDIR=${ZIPDIR:-data/hasmig_mhcs/burial_score_output}
OUT=${OUT:-data/processed/h5_store_hasmig}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

mkdir -p logs/hasmig "$OUT"
[[ -f "$CSV" ]] || { echo "FATAL: $CSV not found"; exit 1; }
[[ -d "$ZIPDIR" ]] || { echo "FATAL: $ZIPDIR not found (untar the tarball first)"; exit 1; }
echo "=== $(date '+%F %T') | hasmig chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} on $(hostname) ==="

$PY src/model_multimer_1/preprocess_hasmig.py \
    --csv "$CSV" --zip-dir "$ZIPDIR" --out-dir "$OUT" \
    --chunk "${SLURM_ARRAY_TASK_ID}" --n-chunks "${N_CHUNKS}"

exit $?
