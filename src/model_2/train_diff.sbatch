#!/bin/bash -l
# ============================================================================
# Train model_2 (MHC-Diff: SE(3)-EGNN diffusion). ONE architecture (no variants).
# Trains on the SIDE-CHAIN store so the torsion loss is active. Resumable.
#
# Submit:        sbatch src/model_2/train_diff.sbatch
# Continue:      just re-submit (--resume picks up last.pt).
# Other split:   edit SCHEME / FOLD below.
# ============================================================================
#SBATCH --job-name=mhcdiff
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/diff/%j.out
#SBATCH --error=logs/diff/%j.err

SCHEME=two_axis
FOLD=1
# ---- optimal for an A100-40GB (peak ~21 GB at bs=32; drop to 16 if a fold has
#      unusually large complexes; k is the main memory knob for the triplet term).
BS=32
LR=2e-4
EPOCHS=60
LAYERS=6
K=12

# Groove-aware loss: dock IN-pocket peptides, expel OUT-of-pocket ones to a [tau_out,
# tau_far] Å shell (containment), keep learning pLDDT on all. Loss-only (no new
# params) -> resumes the current checkpoint in place.
GROOVE_AWARE=0                   # DISABLED (made it worse) — 1 to re-enable
LAMBDA_CONTAIN=0.5
GROOVE="8.0 1.5 8.0 18.0"        # tau_mid  s  tau_out  tau_far  (Å)
GROOVE_FLAG=""
[[ "${GROOVE_AWARE}" == "1" ]] && GROOVE_FLAG="--groove-aware --lambda-contain ${LAMBDA_CONTAIN} --groove ${GROOVE}"

RUN="mhcdiff_${SCHEME}_fold${FOLD}"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/diff "checkpoints_model2/${RUN}"
RUN_LOG="checkpoints_model2/${RUN}/train_${DATE}.log"
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

H5=data/processed/h5_store_sc        # the SIDE-CHAIN store (has teacher_chi)
[[ -f "$H5/index.csv" ]] || { echo "FATAL: $H5/index.csv not found (reprocess first)"; exit 1; }

# Multi-GPU via DistributedDataParallel: one process per GPU on this node.
# BS is PER-GPU (effective batch = BS x #GPUs); each rank gets a data shard so the
# epoch is ~#GPUs times faster. num-workers is per rank -> keep <= cpus/#GPUs.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "launching on ${NGPU} GPU(s) via torchrun"
$PY -m torch.distributed.run --standalone --nproc_per_node=${NGPU} \
    src/model_2/train.py \
    --h5-dir "$H5" --scheme "$SCHEME" --fold "$FOLD" \
    --ckpt-dir checkpoints_model2 --run-name "$RUN" \
    --resume "checkpoints_model2/${RUN}/last.pt" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} --layers ${LAYERS} --k ${K} \
    --lambdas 1.0 0.25 0.5 0.1 \
    ${GROOVE_FLAG} \
    --num-workers 4 --amp

exit $?
