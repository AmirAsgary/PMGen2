#!/bin/bash -l
# ============================================================================
# FAST variant comparison: ONE scheme, FOLD 1 ONLY, all 7 variants = 7 runs.
# Same code/knobs as train_grid.sbatch but a single fold + single split scheme,
# to compare variants (and recycling / unfreezing / peptide-weight) quickly
# before committing to the full 5-fold sweep (train_grid.sbatch).
#
#   array index -> variant 1..7; scheme + fold fixed below
#   each run is isolated under checkpoints/<run>/{config.json,last.pt,best.pt,
#     metrics.csv, tb/, train_<date>.log}
#
# Before first submit:   mkdir -p logs/distill
# Submit:                sbatch src/model/train_grid_1fold.sbatch
# Other scheme/fold:     edit SCHEME / F below (e.g. SCHEME=hla_only)
#
# IDEMPOTENT (re-submit any time): never-started -> fresh; cancelled -> resumes
# from its exact step; finished -> exits with no GPU work; corrupt -> restarts.
# ============================================================================
#SBATCH --job-name=pmgen2_1fold
#SBATCH --array=0-6%7              # 7 variants; %7 = all at once (tune to GPU quota)
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2          # TWO A100s per task (DDP, both GPUs used)
#SBATCH --cpus-per-task=8          # ~num-workers x #GPUs
#SBATCH --mem=128G                 # data pipeline RSS grows over an epoch
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill/%A_%a.out
#SBATCH --error=logs/distill/%A_%a.err

# ---- pick the split (edit here); array index selects the variant ----
SCHEME=two_axis                  # split scheme: two_axis (hardest) or hla_only
F=1                              # fold 1 only
IDX=${SLURM_ARRAY_TASK_ID}
V=$(( IDX + 1 ))                 # variant 1..7
S=${SCHEME}

# ---- hyperparameters (keep in sync with train_grid.sbatch) ----
EPOCHS=12                         # was 6; bump to CONTINUE the rc3 runs longer
BS=10
LR=1.4e-3
PEPTIDE_W=5.0
# AF2-style loss weights: geometry (FAPE) dominates; pLDDT/PAE are light aux heads
# (0.01) so the model spends its capacity on structure, not confidence calibration.
LAMBDAS="1.0 0.01 0.01"          # FAPE  PLDDT  PAE
# Down-weight low-confidence teacher structures in the geometry loss: each example's
# FAPE is scaled by its median peptide pLDDT/100 (floored), so noisy targets count
# less. Loss-only -> does NOT change the run dir / checkpoint -> resumes in place.
PLDDT_WEIGHT_STRUCT=0            # DISABLED (made it worse) — 1 to re-enable
PLDDT_WEIGHT_FLOOR=0.1
# Groove-aware loss: dock IN-pocket peptides exactly, push OUT-of-pocket ones to a
# shell (don't predict their exact pose), keep learning pLDDT on all peptide residues
# so non-binders get flagged. Loss-only -> resumes in place.
GROOVE_AWARE=0                   # DISABLED (made it worse) — 1 to re-enable
LAMBDA_CONTAIN=0.5
GROOVE="8.0 1.5 8.0 18.0"        # tau_mid  s  tau_out  tau_far  (Å)
RECYCLES=3                        # keep =3 so RUN matches the existing _rc3 dirs
# Per-step recycle distribution during training. Empty = uniform 1..RECYCLES.
# "0.8 0.2" = 80% no recycle / 20% one recycle (your lighter schedule).
RECYCLE_PROBS="0.8 0.2"
EVAL_RECYCLES=1                   # validate with 1 recycle
UNFREEZE_SM=0                     # % of each frozen sub-module to unfreeze
UNFREEZE_PLDDT=0
UNFREEZE_PAE=0
ANCHOR_RELPOS=0                   # 1 = encode the alignment register in relpos
                                  # (experimental A/B; fresh run -> _arp dirs)

# ---- run name encodes the non-default knobs (so experiments never collide) ----
RUN="${S}_fold${F}_variant${V}"
[[ "${PEPTIDE_W}" != "1.0" && "${PEPTIDE_W}" != "1" ]] && RUN="${RUN}_pw${PEPTIDE_W}"
[[ "${RECYCLES}" != "0" ]] && RUN="${RUN}_rc${RECYCLES}"
UF="${UNFREEZE_SM}-${UNFREEZE_PLDDT}-${UNFREEZE_PAE}"
[[ "${UF}" != "0-0-0" ]] && RUN="${RUN}_uf${UF}"
ARP_FLAG=""
[[ "${ANCHOR_RELPOS}" == "1" ]] && { ARP_FLAG="--anchor-relpos"; RUN="${RUN}_arp"; }
# pLDDT-weighted structure loss is loss-only (no architecture change) -> the RUN
# name is intentionally NOT modified, so existing _rc3 checkpoints resume in place.
PWS_FLAG=""
[[ "${PLDDT_WEIGHT_STRUCT}" == "1" ]] && PWS_FLAG="--plddt-weight-struct --plddt-weight-floor ${PLDDT_WEIGHT_FLOOR}"
GROOVE_FLAG=""
[[ "${GROOVE_AWARE}" == "1" ]] && GROOVE_FLAG="--groove-aware --lambda-contain ${LAMBDA_CONTAIN} --groove ${GROOVE}"

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

# ---- Multi-GPU via DDP (torchrun): one process per GPU on this task's node.
# BS is PER-GPU (effective batch = BS x #GPUs); each rank gets a disjoint data shard
# so the epoch is ~#GPUs faster. A UNIQUE rendezvous port per array task avoids a
# collision when SLURM co-locates two array tasks on the same node.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
PORT=$((29500 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "launching on ${NGPU} GPU(s) via torchrun (rdzv port ${PORT})"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model/train.py \
    --h5-dir "$H5" \
    --scheme "$S" --fold "$F" --variant "$V" \
    --ckpt-dir checkpoints --run-name "$RUN" \
    --resume "$CKPT" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} --grad-clip 1.0 \
    --lambdas ${LAMBDAS} \
    --peptide-weight ${PEPTIDE_W} --recycles ${RECYCLES} \
    --recycle-probs ${RECYCLE_PROBS} --eval-recycles ${EVAL_RECYCLES} \
    --unfreeze-sm ${UNFREEZE_SM} --unfreeze-plddt ${UNFREEZE_PLDDT} \
    --unfreeze-pae ${UNFREEZE_PAE} ${ARP_FLAG} ${PWS_FLAG} ${GROOVE_FLAG} \
    --log-every 100 --ckpt-every 2000 \
    --num-workers 4 --amp

exit $?
