#!/bin/bash -l
# ============================================================================
# Train model_3 (AF-Evo distill): first K AF2 Evoformer blocks, mostly frozen,
# last block's last TRAINABLE% fine-tuned -> frozen AF2 SM + pLDDT/PAE. Distilled
# to the teacher structures. Resumable; multi-GPU via DDP (torchrun).
#
# Submit:     sbatch src/model_3/train_af3.sbatch
# Continue:   just re-submit (--resume picks up last.pt at the epoch boundary).
# ============================================================================
#SBATCH --job-name=pmgen2_af3
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G                 # the full AF2 model is built per rank at init
#SBATCH --time=24:00:00
#SBATCH --output=logs/af3/%j.out
#SBATCH --error=logs/af3/%j.err

SCHEME=two_axis
FOLD=1
EVO_LAYERS=10                       # K: first AF2 Evoformer blocks to keep (of 48)
TRAINABLE=30                       # % of the LAST kept block to fine-tune
BS=4                               # frozen prefix runs under no_grad -> bs can grow;
                                   #   raise (8/16) while it fits in 40GB (nvidia-smi)
LR=5e-4
EPOCHS=3

RUN="af3_${SCHEME}_fold${FOLD}_K${EVO_LAYERS}"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/af3 "checkpoints_model3/${RUN}"
RUN_LOG="checkpoints_model3/${RUN}/train_${DATE}.log"
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2)
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) ==="
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

H5=data/processed/h5_store
[[ -f "$H5/index.csv" ]] || { echo "FATAL: $H5/index.csv not found"; exit 1; }

# Multi-GPU via DDP; one process per visible GPU; data sharded per rank. BS is
# PER-GPU. Unique rendezvous port avoids a clash if tasks co-locate on a node.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
PORT=$((29600 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "launching on ${NGPU} GPU(s) via torchrun (rdzv port ${PORT})"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model_3/train.py \
    --h5-dir "$H5" --scheme "$SCHEME" --fold "$FOLD" \
    --evo-layers ${EVO_LAYERS} --trainable ${TRAINABLE} \
    --ckpt-dir checkpoints_model3 --run-name "$RUN" \
    --resume "checkpoints_model3/${RUN}/last.pt" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --ckpt-every 1000 \
    --num-workers 4 --amp

exit $?
