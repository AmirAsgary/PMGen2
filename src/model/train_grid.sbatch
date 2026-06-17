#!/bin/bash -l
# ============================================================================
# Train the FULL grid in parallel, one A100 per task, fully resumable.
#
#   schemes (two_axis, hla_only) x folds 1..5 x variants 1..7  = 70 runs
#   array index -> (scheme, fold, variant); each run is isolated under
#     checkpoints/<scheme>_fold<f>_variant<v>/{config.json,last.pt,best.pt}
#   organized logs: logs/distill/<scheme>_fold<f>_variant<v>.{out,err}
#
# Before first submit:   mkdir -p logs/distill
# Submit:                sbatch src/model/train_grid.sbatch
# Continue later:        just re-submit — --resume picks up each run's last.pt.
#
# Subsets via the array range:
#   two_axis only : --array=0-34      hla_only only : --array=35-69
#   one variant   : (variant v) every (idx%7+1)==v       etc.
# ============================================================================
#SBATCH --job-name=pmgen2_grid
#SBATCH --array=0-69%16            # %16 = max concurrent tasks; tune to GPU quota
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1          # ONE A100 per task (training is single-GPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill/%A_%a.out
#SBATCH --error=logs/distill/%A_%a.err

# ---- decode array index -> (scheme, fold, variant) ----
SCHEMES=(two_axis hla_only)
IDX=${SLURM_ARRAY_TASK_ID}
V=$(( IDX % 7 + 1 ))              # variant 1..7
F=$(( (IDX / 7) % 5 + 1 ))        # fold 1..5
S=${SCHEMES[$(( IDX / 35 ))]}     # scheme
RUN="${S}_fold${F}_variant${V}"

# ---- hyperparameters (edit here) ----
# BS is shared by all 7 variants; size it for the heaviest (variant 7). Memory
# scales ~ BS * N^2; class-I pMHC N is small (~190), so a 40 GB A100 has headroom.
# Check variant-7 memory (`ssh <node> nvidia-smi`) and push BS up if it fits;
# scale LR ~ sqrt(BS/8) when you do. Larger BS => far fewer steps/epoch.
EPOCHS=3
BS=16
LR=1.4e-3

# ---- organized, human-readable log symlinks -> this task's slurm files ----
mkdir -p logs/distill "checkpoints/${RUN}"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "logs/distill/${RUN}.out"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "logs/distill/${RUN}.err"
echo "=== task ${IDX}: scheme=${S} fold=${F} variant=${V} -> ${RUN} ==="
nvidia-smi -L || true

export PYTHONUNBUFFERED=1          # stream python stdout/stderr live to the logs
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2"
mamba activate pmgen2

# (optional) stage the store to node-local NVMe for faster epochs:
#   cp -r data/processed/h5_store "$SLURM_TMPDIR/h5_store" && H5=$SLURM_TMPDIR/h5_store
H5=data/processed/h5_store

$PY src/model/train.py \
    --h5-dir "$H5" \
    --scheme "$S" --fold "$F" --variant "$V" \
    --ckpt-dir checkpoints --run-name "$RUN" \
    --resume "checkpoints/${RUN}/last.pt" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} --grad-clip 1.0 \
    --num-workers 8 --amp

exit $?
