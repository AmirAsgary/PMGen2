#!/bin/bash -l
# ============================================================================
# POST-TRAINING DIAGNOSTICS — anchor conditioning + inference latency.
#
# 1. anchor_sensitivity.py — does the model CONDITION ON THE ANCHOR?
#    Each pMHC complex in the store has several AlphaFold structures that differ ONLY
#    in the anchor handed to AlphaFold (100% of the 225,616 multi-structure complexes
#    have all-distinct anchors). Learning anchor -> structure IS the task, so this
#    holds sequence + MHC fixed, changes only the anchor, and compares how far the
#    model moves the peptide against how far AlphaFold moved it.
#    An UNTRAINED control runs first: a random-init model should show ratio ~0, so a
#    trained model that also shows ~0 has learned nothing about the anchor.
#
# 2. bench_latency.py — ms/structure against the 10-50 ms design target, split into
#    the trainable encoder vs the FROZEN AlphaFold decoder (the latter is a floor that
#    no amount of training will lower).
#
#   sbatch --dependency=afterany:<train_job> src/model_multimer_1/run_diagnostics.sh
# ============================================================================
#SBATCH --job-name=mm1_diag
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

RUN=${RUN:-mm1_s1_short}
CKPT=${CKPT:-checkpoints_mm1/${RUN}/best.pt}
[[ -f "$CKPT" ]] || CKPT=checkpoints_mm1/${RUN}/last.pt

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2

echo "=== $(date '+%F %T') | diagnostics on $(hostname) | ckpt=$CKPT ==="
[[ -f "$CKPT" ]] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }

echo
echo "############ 1a. ANCHOR SENSITIVITY — UNTRAINED CONTROL ############"
$PY src/model_multimer_1/anchor_sensitivity.py --n 120 --pep-frames identity

echo
echo "############ 1b. ANCHOR SENSITIVITY — TRAINED ############"
$PY src/model_multimer_1/anchor_sensitivity.py --ckpt "$CKPT" --n 120 --pep-frames identity

echo
echo "############ 2. INFERENCE LATENCY (target 10-50 ms/structure) ############"
$PY src/model_multimer_1/bench_latency.py --ckpt "$CKPT" --batch 1 4 8 --iters 30

echo "=== $(date '+%F %T') | diagnostics done ==="
