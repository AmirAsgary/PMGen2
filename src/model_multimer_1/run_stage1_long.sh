#!/bin/bash -l
# ============================================================================
# STAGE 1 — LONG CONTINUATION from the best pre-collapse checkpoint.
#
# Starts from g29600, NOT the newest file: g29600 scores 2.73 A full-atom peptide
# RMSD on data/test vs 2.90 A for g52818, which came from the run that was already
# degrading. Newest != best here.
#
# ROOT-CAUSE FIX (--angle-input layernorm, now the default): the frozen SM LayerNorms
# its input internally, so the STRUCTURE loss is scale-invariant w.r.t. sm_in and nothing
# penalised the trunk for inflating it. The trainable AngleResnet was the ONLY consumer
# seeing the RAW scale (AlphaFold feeds it the POST-LayerNorm single; we fed it the raw
# projection), and it adds that branch straight into its residual stream. MEASURED across
# checkpoints: ||raw|| 2393 -> 14695 -> 16325 -> 22667 (9.5x, unbounded, monotonic) while
# ||LayerNormed|| stays 261 -> 185. The angle head had been ADAPTING to the inflation —
# switching those checkpoints to normalised input costs 0.375 -> 0.781 loss in proportion
# to how far each had drifted. Its gradients flow back into the SHARED trunk, which is why
# the MHC (an INPUT, normally reproduced to 0.16 A) collapsed together with the peptide
# while loss, |g| and loss-landscape sharpness ALL still looked healthy.
#
# NOTE: resuming from a checkpoint trained with the RAW convention means the angle head
# must re-adapt — expect chi/sc_fape to start high (~0.67 total vs ~0.30) and come down.
#
# WHY lr 3e-4 and not 5e-4: three runs have died at 5e-4 (July ~24.8k, 29377637 at
# 32.0k, 29383715 at 54.2k). The telemetry shows the STRUCTURE collapses first with
# |g| still at 5-6 -- the gradient explosion is FAPE evaluated on already-destroyed
# geometry, not the cause. The one run that never diverged is the short one, whose
# cosine T_max was 9k so the LR actually came down; the full runs hold ~4.9e-4 for
# tens of thousands of steps. Lower LR is a MITIGATION of a structural weakness (the
# frozen SM must rebuild all 190 residues from sm_s(trunk_single) with nothing keeping
# that vector in-distribution), not a cure. Job 29385773 is the running control at
# 5e-4 from the same weights.
#
# --fresh-optim so the new LR actually takes effect (a plain --resume restores the old
# optimizer AND scheduler state, which would silently reinstate 5e-4).
#
# CHAINING: 6 epochs ~= 39 h at 4 GPUs against a 24 h cap. last.pt holds the last
# COMPLETED epoch; link 2+ resumes it. Pass args POSITIONALLY -- `VAR=x sbatch` does
# not reliably reach the batch job and silently falls back to the default.
#
#   sbatch src/model_multimer_1/run_stage1_long.sh                    # link 1
#   sbatch src/model_multimer_1/run_stage1_long.sh auto               # link 2+
# ============================================================================
#SBATCH --job-name=mm1_s1_long
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=23:30:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

RUN=${2:-mm1_s1_long}
RESUME=${1:-checkpoints_mm1/mm1_s1_full_best_g29600.pt}
LR=${3:-3e-4}
EPOCHS=${4:-6}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
mkdir -p logs/mm1

FRESH="--fresh-optim"
if [[ "$RESUME" == "auto" ]]; then                 # continuing OUR OWN chain
    RESUME="checkpoints_mm1/${RUN}/last.pt"
    FRESH=""                                       # keep optimizer+schedule continuous
    [[ -f "$RESUME" ]] || { echo "FATAL: chain link asked for but $RESUME missing"; exit 1; }
fi
[[ -f "$RESUME" ]] || { echo "FATAL: $RESUME missing"; exit 1; }

$PY - "$RESUME" <<'PYCHK' || exit 1
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("trainable", {})
bad = [k for k, v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
if bad:
    print(f"FATAL: {sys.argv[1]} has {len(bad)}/{len(sd)} non-finite tensors — refusing.")
    sys.exit(1)
print(f"  ok: {sys.argv[1]} finite ({len(sd)} tensors)")
PYCHK

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU | lr=${LR} ==="
echo "    resume=${RESUME} ${FRESH}  epochs=${EPOCHS}"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29833 \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir data/processed/h5_store_sc --hasmig-dir data/processed/h5_store_hasmig \
    --data-exp-csv outputs/data_exploration/per_structure.csv \
    --scheme two_axis --fold 1 \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
    --max-val 3000 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --resume "$RESUME" $FRESH \
    --epochs ${EPOCHS} --bs 1 --lr ${LR} \
    --warmup-steps 500 --divergence-factor 8.0 \
    --log-every 100 --train-metrics-every 25 --ckpt-every 1000 \
    --num-workers 6 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
