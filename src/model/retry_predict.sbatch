#!/bin/bash -l
# ============================================================================
# Re-run PMGen for ONLY the failed ids, balanced across jobs, written back into
# each id's own chunk output dir. Generate the manifest first:
#
#   python src/model/make_retry_inputs.py \
#       --chunks-dir ~/projects/PMGen_2/data/pmgen_inputs/chunks \
#       --h5-dir     data/processed/h5_store \
#       --out-dir    data/processed/retry_inputs --max-jobs 32
#
# then set --array=1-<N> below to the N it prints and:  sbatch src/model/retry_predict.sbatch
# ============================================================================
#SBATCH -o ./logs/retry.%A_%a.out
#SBATCH -e ./logs/retry.%A_%a.err
#SBATCH -J pmgen_retry
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --time=06:00:00
#SBATCH --array=1-32          # <-- set to the N printed by make_retry_inputs.py

CHUNKS_DIR=/u/${USER}/projects/PMGen_2/data/pmgen_inputs/chunks
MANIFEST=$HOME/projects/PMGen_2/PMGen2/data/processed/retry_inputs/manifest.tsv

# this task's row (skip header): cols = shard_id chunk tsv n_ids output_dir
ROW=$(awk -v i=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==i' "$MANIFEST")
CHUNK=$(echo "$ROW"  | cut -f2)
TSV=$(echo "$ROW"    | cut -f3)
OUTPUT_DIR=$(echo "$ROW" | cut -f5)
RETRY_DIR=$(dirname "$TSV")
echo "shard ${SLURM_ARRAY_TASK_ID}: chunk=$CHUNK n=$(echo "$ROW"|cut -f4) out=$OUTPUT_DIR tsv=$TSV"
[ -n "$TSV" ] || { echo "no manifest row for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

mkdir -p "$OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "${CHUNKS_DIR}/${CHUNK}/output"

# OPTIONAL — if PMGen skips ids whose output folder already exists, clear the
# incomplete dirs for THIS shard's ids first (they hold no usable output):
#   awk -F'\t' -v c="$CHUNK" 'NR>1 && $2==c {print}' "$MANIFEST" >/dev/null  # (per-shard)
#   while read -r id; do rm -rf "$OUTPUT_DIR/alphafold/$id"; done < <(tail -n +2 "$TSV" | cut -f5)

module purge
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
CUDA_BIN=$(dirname $(which ptxas))
SIF=$HOME/containers/pmgen.sif

apptainer exec --nv \
  --bind $CUDA_BIN:/host_cuda_bin \
  --env PATH=/host_cuda_bin:/opt/conda/envs/PMGen/bin:/usr/bin:/bin \
  --bind /ptmp/${USER}/pmgen_run:/ptmp/${USER}/pmgen_run \
  --bind ${CHUNKS_DIR}:${CHUNKS_DIR} \
  --bind ${RETRY_DIR}:${RETRY_DIR} \
  $SIF \
  bash -c "cd /opt/PMGen && python run_PMGen.py \
    --mode wrapper --run parallel \
    --df ${TSV} \
    --output_dir ${OUTPUT_DIR} \
    --initial_guess"

exit $?
