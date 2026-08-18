#!/bin/bash -l
# ============================================================================
# DIVERGENCE PROBE — is the collapse driven by the LEARNING RATE?
#
# Evidence it is NOT gradient spikes (from run 29383715, which HAD spike rejection):
#   gstep 53918 -> 54118, |g|max stayed 5.6 / 4.8 / 6.0 (perfectly normal) while
#   ca_rmsd climbed 0.587 -> 0.714 -> 1.388 and pep_rmsd 2.29 -> 3.02 -> 5.29.
#   The model degraded SMOOTHLY under ordinary gradients. The "50 consecutive
#   rejected spikes" that aborted it came only afterwards, once the running median
#   had collapsed. Rejecting spikes therefore treats a symptom.
#
# Both deaths happened with the LR pinned near peak (cosine T_max = 307k steps, so
# lr ~4.9e-4 for tens of thousands of steps). The ONLY run that never diverged is
# the short one, whose T_max was 9k so the LR actually came down.
#
# This resumes from the SAME pre-collapse weights with a FRESH optimizer, varying
# ONLY the LR. Run 29383715 reproduced the collapse ~2,950 steps after resuming from
# this checkpoint, so ~45 min at 2 GPUs is enough to see it.
#   arm A: lr 5e-4  (the setting that died -> expect collapse)
#   arm B: lr 1e-4  (hypothesis -> expect survival)
# --train-metrics-every 10 so the behavioural collapse is visible at 10-step
# resolution instead of 100 (that is how ca_rmsd 16.3 stayed invisible until too late).
#
#   sbatch src/model_multimer_1/run_lr_probe.sh <lr> <tag>
# ============================================================================
#SBATCH --job-name=mm1_lrprobe
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

LR=${1:?usage: sbatch run_lr_probe.sh <lr> <tag>}
TAG=${2:?usage: sbatch run_lr_probe.sh <lr> <tag>}
RESUME=${3:-checkpoints_mm1/mm1_s1_full_best_g29600.pt}
RUN="mm1_lrprobe_${TAG}"

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
mkdir -p logs/mm1

[[ -f "$RESUME" ]] || { echo "FATAL: $RESUME missing"; exit 1; }
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU | lr=${LR} ==="
echo "    resume=${RESUME} (FRESH optimizer: only the LR differs between arms)"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=298${RANDOM:0:2} \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir data/processed/h5_store_sc --hasmig-dir data/processed/h5_store_hasmig \
    --data-exp-csv outputs/data_exploration/per_structure.csv \
    --scheme two_axis --fold 1 \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
    --max-val 1000 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --resume "$RESUME" --fresh-optim \
    --epochs 6 --bs 1 --lr ${LR} \
    --warmup-steps 200 --divergence-factor 8.0 --grad-spike-factor 0 \
    --log-every 50 --train-metrics-every 10 --ckpt-every 2000 \
    --num-workers 6 --amp

echo "=== $(date '+%F %T') | ${RUN} exit $? ==="
