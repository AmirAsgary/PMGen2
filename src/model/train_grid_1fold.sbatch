#!/bin/bash -l
# ============================================================================
# FAST variant comparison: FOLD 1 ONLY, all variants, both schemes = 14 runs.
# Same code/knobs as train_grid.sbatch but a single fold, to compare variants
# (and recycling / unfreezing / peptide-weight) quickly before committing to the
# full 5-fold sweep. Run the 5-fold version (train_grid.sbatch) afterwards.
#
#   array index -> (scheme, variant), fold fixed to 1  (2 schemes x 7 variants)
#   each run is isolated under checkpoints/<run>/{config.json,last.pt,best.pt,
#     metrics.csv, tb/, train_<date>.log}
#
# Before first submit:   mkdir -p logs/distill
# Submit:                sbatch src/model/train_grid_1fold.sbatch
# Subsets: two_axis only --array=0-6 ; hla_only only --array=7-13
#
# IDEMPOTENT (re-submit any time): never-started -> fresh; cancelled -> resumes
# from its exact step; finished -> exits with no GPU work; corrupt -> restarts.
# ============================================================================
#SBATCH --job-name=pmgen2_1fold
#SBATCH --array=0-13%14            # 14 runs; %14 = all at once (tune to GPU quota)
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2          # ONE A100 per task (training is single-GPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G                 # data pipeline RSS grows over an epoch
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill/%A_%a.out
#SBATCH --error=logs/distill/%A_%a.err

# ---- decode array index -> (scheme, variant); fold is fixed to 1 ----
SCHEMES=(two_axis hla_only)
IDX=${SLURM_ARRAY_TASK_ID}
V=$(( IDX % 7 + 1 ))              # variant 1..7
S=${SCHEMES[$(( IDX / 7 ))]}      # scheme (idx 0-6 two_axis, 7-13 hla_only)
F=1                              # FOLD 1 ONLY

# ---- hyperparameters (keep in sync with train_grid.sbatch) ----
EPOCHS=6
BS=10
LR=1.4e-3
PEPTIDE_W=5.0
RECYCLES=3                        # eval uses N; training samples 1..N per step
UNFREEZE_SM=0                     # % of each frozen sub-module to unfreeze
UNFREEZE_PLDDT=0
UNFREEZE_PAE=0

# ---- run name encodes the non-default knobs (so experiments never collide) ----
RUN="${S}_fold${F}_variant${V}"
[[ "${PEPTIDE_W}" != "1.0" && "${PEPTIDE_W}" != "1" ]] && RUN="${RUN}_pw${PEPTIDE_W}"
[[ "${RECYCLES}" != "0" ]] && RUN="${RUN}_rc${RECYCLES}"
UF="${UNFREEZE_SM}-${UNFREEZE_PLDDT}-${UNFREEZE_PAE}"
[[ "${UF}" != "0-0-0" ]] && RUN="${RUN}_uf${UF}"

DATE=$(date +%Y%m%d_%H%M%S)
RUN_LOG="checkpoints/${RUN}/train_${DATE}.log"

mkdir -p logs/distill "checkpoints/${RUN}"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "logs/distill/${RUN}.out"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "logs/distill/${RUN}.err"
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2)

echo "=== $(date '+%F %T') | task ${IDX}: ${RUN} | log ${RUN_LOG} ==="
echo "=== job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
nvidia-smi -L || true

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

H5=data/processed/h5_store
[[ -f "$H5/index.csv" ]] || { echo "FATAL: H5 store '$H5/index.csv' not found"; exit 1; }

CKPT="checkpoints/${RUN}/last.pt"
if [[ -f "$CKPT" ]]; then echo "resume: found ${CKPT} (continue / skip-if-done)";
else echo "resume: no checkpoint -> fresh start"; fi

$PY src/model/train.py \
    --h5-dir "$H5" \
    --scheme "$S" --fold "$F" --variant "$V" \
    --ckpt-dir checkpoints --run-name "$RUN" \
    --resume "$CKPT" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} --grad-clip 1.0 \
    --peptide-weight ${PEPTIDE_W} --recycles ${RECYCLES} \
    --unfreeze-sm ${UNFREEZE_SM} --unfreeze-plddt ${UNFREEZE_PLDDT} \
    --unfreeze-pae ${UNFREEZE_PAE} \
    --log-every 100 --ckpt-every 2000 \
    --num-workers 4 --amp

exit $?
