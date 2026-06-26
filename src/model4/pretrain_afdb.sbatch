#!/bin/bash -l
# ============================================================================
# model4 / plddt_model — stage-1 pretraining on the AFDB monomer set.
# Predicts per-residue pLDDT (50-bin CE) + aux masked-LM + aux anchor/contact.
# AF pLDDT head initialised + frozen for FREEZE_PLDDT epochs, then unfrozen.
# Resumable; multi-GPU via DDP (torchrun).
#
# Submit:   sbatch src/model4/pretrain_afdb.sbatch
# Continue: re-submit (--resume picks up last.pt at the epoch boundary).
# ============================================================================
#SBATCH --job-name=pmgen2_m4
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=280G
#SBATCH --time=24:00:00
#SBATCH --output=logs/m4_2/%j.out
#SBATCH --error=logs/m4_2/%j.err

# Env-overridable knobs (no file edit for a parallel variant).
BLOCKS=${BLOCKS:-1}
CROP=${CROP:-256}
CS=${CS:-32}                        # body single channel (AF pLDDT head up-projects to 384)
CZ=${CZ:-64}                        # body pair channel
BS=${BS:-16}
LR=${LR:-1e-2}
# pLDDT head: frozen epoch 0, then 1 layer/epoch unfrozen (output->input) over the
# next 4 epochs -> keep EPOCHS >= FREEZE_PLDDT+4 so it fully unfreezes with margin.
EPOCHS=${EPOCHS:-8}
# Per-epoch validation batches PER RANK (val set is huge: ~230k). Distributed eval
# shards across ranks; a cap keeps each epoch's val quick. Set EVAL_BATCHES=0 for full val.
EVAL_BATCHES=${EVAL_BATCHES:-200}
DROPOUT=${DROPOUT:-0.1}
MLM_FRAC=${MLM_FRAC:-0.15}
ANCHOR_REVEAL=${ANCHOR_REVEAL:-0.2}
ANCHOR_SEQSEP=${ANCHOR_SEQSEP:-6}
FREEZE_PLDDT=${FREEZE_PLDDT:-1}
# GRADUAL=1: unfreeze the pLDDT head one layer/epoch after the freeze (default).
# GRADUAL=0: unfreeze the WHOLE head at once (use with FREEZE_PLDDT=0 to continue a
#            run with the full head trainable from the resumed epoch).
GRADUAL=${GRADUAL:-1}
GRADUAL_FLAG="--gradual-unfreeze"; [[ "${GRADUAL}" == "0" ]] && GRADUAL_FLAG="--no-gradual-unfreeze"
WARMUP=${WARMUP:-1000}
# RESET_SCHED=1: on resume, start a FRESH optimizer + warmup->cosine over the
# remaining steps using the LR/WARMUP below (e.g. to fine-tune at a lower LR).
RESET_SCHED=${RESET_SCHED:-0}
RESET_FLAG=""; [[ "${RESET_SCHED}" == "1" ]] && RESET_FLAG="--reset-schedule"
L1=${L1:-0}
WD=${WD:-1e-4}
LAMBDAS=${LAMBDAS:-"1.0 0.1 0.1 0.1"}    # plddt mlm anchor distogram
# GRADCKPT=1 trades a little compute for ~40% less activation memory -> bigger BS.
GRADCKPT=${GRADCKPT:-0}
GC_FLAG=""; [[ "${GRADCKPT}" == "1" ]] && GC_FLAG="--grad-checkpoint"

RUN="afdb_b${BLOCKS}_crop${CROP}_2"
[[ "${L1}" != "0" ]] && RUN="${RUN}_l1${L1}"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/m4 "checkpoints_model4/${RUN}"
RUN_LOG="checkpoints_model4/${RUN}/train_${DATE}.log"
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2)
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) ==="
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || cd "$HOME/amir/ParseFold/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

H5=data/afdb/plddt_dataset.h5
DOMAINS=data/afdb/final_domains_mapped_clean.tsv
[[ -f "$H5" ]] || { echo "FATAL: $H5 not found"; exit 1; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
PORT=$((29700 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "launching on ${NGPU} GPU(s) via torchrun (rdzv port ${PORT})"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model4/train.py \
    --h5 "$H5" --domains-tsv "$DOMAINS" \
    --blocks ${BLOCKS} --crop-len ${CROP} --c-s ${CS} --c-z ${CZ} --dropout ${DROPOUT} \
    --mlm-frac ${MLM_FRAC} --anchor-reveal-prob ${ANCHOR_REVEAL} \
    --anchor-seq-sep ${ANCHOR_SEQSEP} --freeze-plddt-epochs ${FREEZE_PLDDT} \
    ${GRADUAL_FLAG} --warmup-steps ${WARMUP} ${GC_FLAG} ${RESET_FLAG} --eval-batches ${EVAL_BATCHES} \
    --lambdas ${LAMBDAS} --l1 ${L1} --weight-decay ${WD} \
    --ckpt-dir checkpoints_model4 --run-name "$RUN" \
    --resume "checkpoints_model4/${RUN}/last.pt" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --ckpt-every 1000 \
    --num-workers 4 --amp

exit $?
