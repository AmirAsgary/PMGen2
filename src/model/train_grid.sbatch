#!/bin/bash -l
# ============================================================================
# Train the FULL grid in parallel, one A100 per task, fully resumable.
#
#   schemes (two_axis, hla_only) x folds 1..5 x variants 1..7  = 70 runs
#   array index -> (scheme, fold, variant); each run is isolated under
#     checkpoints/<scheme>_fold<f>_variant<v>/
#       {config.json, last.pt, best.pt, metrics.csv, tb/, train_<date>.log}
#   the dated train_<date>.log holds that submission's full output (one per
#   (re)submit). logs/distill/<run>.{out,err} also symlink to the SLURM files.
#
# Before first submit:   mkdir -p logs/distill
# Submit:                sbatch src/model/train_grid.sbatch
#
# IDEMPOTENT: re-submit the whole array any time. Per run, automatically:
#   * never started   -> trains from scratch
#   * cancelled mid-run-> resumes from its exact step (last.pt every 2000 steps)
#   * finished         -> exits immediately, no GPU work (raise EPOCHS to extend)
#   * corrupt last.pt  -> warns and restarts that one run from scratch
# So you never need to hand-pick which tasks to rerun.
#
# Subsets via the array range:
#   two_axis only : --array=0-34      hla_only only : --array=35-69
#   one variant   : (variant v) every (idx%7+1)==v       etc.
# ============================================================================
#SBATCH --job-name=pmgen2_grid
#SBATCH --array=0-69%16            # %16 = max concurrent tasks; tune to GPU quota
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2          # ONE A100 per task (training is single-GPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G                 # data pipeline RSS grows over an epoch (74 GB
                                   # store + worker copies); 64 GB OOM-killed it
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill/%A_%a.out
#SBATCH --error=logs/distill/%A_%a.err

# ---- decode array index -> (scheme, fold, variant) ----
SCHEMES=(two_axis hla_only)
IDX=${SLURM_ARRAY_TASK_ID}
V=$(( IDX % 7 + 1 ))              # variant 1..7
F=$(( (IDX / 7) % 5 + 1 ))        # fold 1..5
S=${SCHEMES[$(( IDX / 35 ))]}     # scheme
# ---- hyperparameters (edit here) ----
# BS is shared by all 7 variants; size it for the heaviest (variant 7). Memory
# scales ~ BS * N^2; class-I pMHC N is small (~190), so a 40 GB A100 has headroom.
# Check variant-7 memory (`ssh <node> nvidia-smi`) and push BS up if it fits;
# scale LR ~ sqrt(BS/8) when you do. Larger BS => far fewer steps/epoch.
EPOCHS=6
BS=10
LR=1.4e-3
# Peptide loss up-weighting (1.0 = uniform/unchanged; ~3-5 emphasizes the peptide).
PEPTIDE_W=5.0
# Recycling: eval uses RECYCLES iterations, training samples 1..RECYCLES per step.
# 0 = no recycling. >0 changes the architecture (adds a recycling embedder), so
# such runs land in their own *_rcN dirs (see SUFFIX below) and are ~1.3-1.7x
# slower to train (extra passes are forward-only).
RECYCLES=3
# Partial fine-tuning of the FROZEN stack: % of each sub-module to unfreeze
# (last params first). 0 = fully frozen (default). Whole tensors are unfrozen,
# so the realised % can exceed the request — see config.json 'unfrozen_fraction'.
UNFREEZE_SM=0
UNFREEZE_PLDDT=0
UNFREEZE_PAE=0

# ---- run name encodes the non-default knobs so different experiments never
#      collide / resume each other. baseline (pw1, rc0, no unfreeze) keeps the
#      plain name so older runs still resume.
RUN="${S}_fold${F}_variant${V}"
[[ "${PEPTIDE_W}" != "1.0" && "${PEPTIDE_W}" != "1" ]] && RUN="${RUN}_pw${PEPTIDE_W}"
[[ "${RECYCLES}" != "0" ]] && RUN="${RUN}_rc${RECYCLES}"
UF="${UNFREEZE_SM}-${UNFREEZE_PLDDT}-${UNFREEZE_PAE}"
[[ "${UF}" != "0-0-0" ]] && RUN="${RUN}_uf${UF}"

# per-submission dated log kept INSIDE the run's checkpoint dir (one file per
# (re)submit, so you get a history: fresh run, resume 1, resume 2, ...)
DATE=$(date +%Y%m%d_%H%M%S)
RUN_LOG="checkpoints/${RUN}/train_${DATE}.log"

# ---- organized, human-readable log symlinks -> this task's slurm files ----
mkdir -p logs/distill "checkpoints/${RUN}"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "logs/distill/${RUN}.out"
ln -sfn "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "logs/distill/${RUN}.err"

# Mirror everything (stdout+stderr) into the dated per-run log in the ckpt dir,
# live (so even a cancelled run keeps its partial log there). The SLURM .out/.err
# files still receive their streams unchanged.
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2)

echo "=== $(date '+%F %T') | task ${IDX}: ${RUN} | log ${RUN_LOG} ==="
echo "=== job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
nvidia-smi -L || true

export PYTHONUNBUFFERED=1          # stream python stdout/stderr live to the logs
# Bound DataLoader-worker RSS growth: glibc fragments badly with many arenas +
# variable-length array reads, and never returns memory to the OS. These cap the
# arenas and force trimming, which is what stops the slow climb to the mem limit.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

# (optional) stage the store to node-local NVMe for faster epochs:
#   cp -r data/processed/h5_store "$SLURM_TMPDIR/h5_store" && H5=$SLURM_TMPDIR/h5_store
H5=data/processed/h5_store
[[ -f "$H5/index.csv" ]] || { echo "FATAL: H5 store '$H5/index.csv' not found"; exit 1; }

# Report this run's state (purely informational; train.py decides what to do):
CKPT="checkpoints/${RUN}/last.pt"
if [[ -f "$CKPT" ]]; then echo "resume: found ${CKPT} (continue / skip-if-done)";
else echo "resume: no checkpoint -> fresh start"; fi

# train.py is idempotent across re-submits:
#   * no last.pt          -> trains from scratch
#   * last.pt mid-run     -> resumes from its exact global_step
#   * last.pt at >=EPOCHS -> prints "nothing to do" and exits 0 (no GPU work)
#   * corrupt last.pt     -> warns and restarts that run from scratch
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
