#!/bin/bash -l
# ============================================================================
# Train model_multimer_1 — leak-free (--pep-frames identity) + AlphaFold-style
# side chains (--sidechains). Three stages; pass STAGE (and optional RESUME) as
# POSITIONAL ARGS ($1/$2) — SLURM forwards them reliably, unlike submit-time env
# vars, which are NOT propagated to the batch job on every cluster.
#
#   STAGE 1  structure ONLY: bb-FAPE 0.5 + sc-FAPE 0.5 + chi 1.0, lambda_plddt = 0
#            (plddt_proj frozen). Confidence-filtered subset; hasmig full weight.
#   STAGE 2  same structure losses + lambda_plddt 0.01 (plddt_proj trainable);
#            hasmig down-weighted 0.1. RESUME from stage-1.  (A/B data variants:
#            use stage2_mm1.sbatch instead.)
#   STAGE 3  confidence ONLY: freeze all but plddt_proj; structure detached;
#            lambda_plddt 1.0 (sc/bb weights auto-zero). RESUME from stage-2.
#
#   sbatch src/model_multimer_1/train_mm1.sbatch 1
#   sbatch src/model_multimer_1/train_mm1.sbatch 2
#   sbatch src/model_multimer_1/train_mm1.sbatch 3 checkpoints_mm1/mm1_stage2_sc/last.pt
#
# PREREQUISITE: BOTH stores preprocessed with side-chain targets, or the job aborts:
#   sbatch src/model_multimer_1/preprocess_hasmig.sbatch   # hasmig  (--sidechains)
#   sbatch src/model/reprocess_sidechains.sbatch           # old two_axis store -> h5_store_sc
# ============================================================================
#SBATCH --job-name=mm1
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

STAGE=${1:-${STAGE:-1}}                    # positional $1 preferred, then env, then 1
RESUME=${2:-${RESUME:-}}                   # positional $2 preferred, then env
H5=${H5:-data/processed/h5_store_sc}       # OLD two_axis store WITH sidechains (val source)
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
SCHEME=${SCHEME:-two_axis}
FOLD=${FOLD:-1}
EPOCHS=${EPOCHS:-6}
BS=${BS:-1}
LR=${LR:-5e-4}                             # 2e-3 -> NaN by ep 15; 5e-4 verified stable
RUN="mm1_stage${STAGE}_sc"

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1 "checkpoints_mm1/${RUN}"

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing (run download+extract weights first)"; exit 1; }
done

RESUME_FLAG=""
[[ -n "$RESUME" ]] && RESUME_FLAG="--resume $RESUME"
# stage 2/3 with no explicit resume: default to the previous stage's checkpoint
if [[ "$STAGE" == "2" && -z "$RESUME" && -f "checkpoints_mm1/mm1_stage1_sc/last.pt" ]]; then
    RESUME_FLAG="--resume checkpoints_mm1/mm1_stage1_sc/last.pt"
fi
if [[ "$STAGE" == "3" && -z "$RESUME" && -f "checkpoints_mm1/mm1_stage2_sc/last.pt" ]]; then
    RESUME_FLAG="--resume checkpoints_mm1/mm1_stage2_sc/last.pt"
fi

# AF2 side-chain losses (bb-FAPE 0.5 / sc-FAPE 0.5 / chi 1.0); leak-free peptide frames.
# In stage 3 train.py auto-zeros the structure weights, so --sidechains is harmless there
# (it just checks the targets exist).
SC_FLAGS="--pep-frames identity --sidechains --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
PORT=$((29700 + STAGE))
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU(s) | lr=${LR} ==="
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model_multimer_1/train.py \
    --stage "$STAGE" --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme "$SCHEME" --fold "$FOLD" $SC_FLAGS \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" $RESUME_FLAG \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
