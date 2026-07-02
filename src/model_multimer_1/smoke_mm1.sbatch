#!/bin/bash -l
# ============================================================================
# MULTI-GPU SMOKE for model_multimer_1 on Raven.
#
# Trains + validates on ONLY 10 randomly-sampled complexes from the hasmig H5
# store, for 2 epochs, across 2 GPUs. Purpose: prove that BOTH training (DDP,
# sharded loader) AND validation (sharded + example-weighted all-reduce) run
# end-to-end on multiple GPUs before committing to a full run. It is NOT meant
# to learn anything — expect the loss to barely move in 2 epochs.
#
#   sbatch src/model_multimer_1/smoke_mm1.sbatch
#
# If this finishes with two "[mm1] val total ..." lines and exit 0, the
# multi-GPU path is good and you can launch train_mm1.sbatch for real.
# ============================================================================
#SBATCH --job-name=mm1_smoke
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/smoke_%j.out
#SBATCH --error=logs/mm1/smoke_%j.err

HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
H5=${H5:-/nonexistent_old_store}          # use ONLY the hasmig store for this smoke
N=${N:-10}                                # #random structures
EPOCHS=${EPOCHS:-2}
BS=${BS:-1}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1 checkpoints_mm1/mm1_smoke

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing (run download+extract weights first)"; exit 1; }
done
[[ -f "$HASMIG/index.csv" ]] || { echo "FATAL: $HASMIG/index.csv missing (preprocess first)"; exit 1; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-2}
echo "=== $(date '+%F %T') | mm1 SMOKE on $(hostname) | ${NGPU} GPU(s) | N=${N} epochs=${EPOCHS} ==="
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29765 \
    src/model_multimer_1/train.py \
    --stage 1 --no-filter --max-train ${N} \
    --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv /nonexistent.csv \
    --ckpt-dir checkpoints_mm1 --run-name mm1_smoke \
    --epochs ${EPOCHS} --bs ${BS} --lr 1e-3 --amp \
    --log-every 1 --ckpt-every 100000 --num-workers 4

RC=$?
echo "=== $(date '+%F %T') | mm1 SMOKE exit ${RC} ==="
exit $RC
