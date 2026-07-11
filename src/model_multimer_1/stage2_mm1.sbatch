#!/bin/bash -l
# ============================================================================
# STAGE 2 (leak-free + side chains) from the stage-1 checkpoint — two data
# variants, meant to be submitted IN PARALLEL. Pass the variant as $1 (A or B);
# a SLURM positional arg is forwarded reliably, unlike a submit-time env var.
#
#   A  continue on the FILTERED structures; validate on the FILTERED val subset.
#      (--force-filter --filter-val)                        -> mm1_stage2_A_sc
#   B  FULL dataset, no filter; structural (bb+sc) FAPE per-structure quality-
#      weighted w_n = w_min + (1-w_min)*q_plddt*q_burial ; validate on ALL.
#      (--struct-quality-weight)                            -> mm1_stage2_B_sc
#
# Both: leak-free peptide frames, AF2 side-chain losses (bb-FAPE .5 / sc-FAPE .5 /
# chi 1.0), pLDDT weight 0.01, hasmig down-weighted 0.1, resumed (weights-only,
# fresh optim) from a preserved copy of stage-1's LAST checkpoint so A and B share
# the exact same init.
#
#   sbatch src/model_multimer_1/stage2_mm1.sbatch A
#   sbatch src/model_multimer_1/stage2_mm1.sbatch B
#
# Two parallel jobs at gpu:a100:4 need 8 GPUs; drop --gres to gpu:a100:2 if your
# quota is 4, or submit B after A.
# ============================================================================
#SBATCH --job-name=mm1_s2
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

APPROACH=${1:-${APPROACH:-}}               # positional $1; no silent default
H5=${H5:-data/processed/h5_store_sc}       # OLD two_axis store WITH sidechains
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
SCHEME=${SCHEME:-two_axis}
FOLD=${FOLD:-1}
EPOCHS=${EPOCHS:-6}
BS=${BS:-1}
LR=${LR:-5e-4}
PLDDT_W=${PLDDT_W:-0.01}
HASMIG_W=${HASMIG_W:-0.1}
WMIN=${WMIN:-0.05}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

# --- preserve stage-1's LAST checkpoint once (A and B share the exact same init) ----
PRETRAINED=checkpoints_mm1/pretrained_stage1_sc_last.pt
SRC=checkpoints_mm1/mm1_stage1_sc/last.pt
if [[ ! -f "$PRETRAINED" ]]; then
    [[ -f "$SRC" ]] || { echo "FATAL: $SRC missing (run stage 1 first)"; exit 1; }
    cp "$SRC" "${PRETRAINED}.tmp.$$" && mv -n "${PRETRAINED}.tmp.$$" "$PRETRAINED"
    rm -f "${PRETRAINED}.tmp.$$"
    echo "preserved stage-1 LAST checkpoint -> $PRETRAINED"
fi

case "$APPROACH" in
  A) FLAGS="--force-filter --filter-val";           RUN=mm1_stage2_A_sc; PORT=29801 ;;
  B) FLAGS="--struct-quality-weight --w-min $WMIN";  RUN=mm1_stage2_B_sc; PORT=29802 ;;
  *) echo "FATAL: approach must be A or B (got '${APPROACH}'). Pass it as the first"\
          "argument, e.g.: sbatch src/model_multimer_1/stage2_mm1.sbatch B"; exit 1 ;;
esac

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing"; exit 1; }
done

# leak-free peptide frames + AF2 side-chain losses (same as stage 1); --fresh-optim so the
# stage-1 -> stage-2 transition restarts the optimizer + a fresh cosine over EPOCHS.
SC_FLAGS="--pep-frames identity --sidechains --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} (Approach ${APPROACH}) on $(hostname) | ${NGPU} GPU(s) ==="
echo "    resume=${PRETRAINED} flags='${FLAGS}' plddt_w=${PLDDT_W} hasmig_w=${HASMIG_W} lr=${LR}"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model_multimer_1/train.py \
    --stage 2 --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme "$SCHEME" --fold "$FOLD" --resume "$PRETRAINED" --fresh-optim \
    $SC_FLAGS --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 \
    --plddt-w "$PLDDT_W" --hasmig-weight "$HASMIG_W" $FLAGS \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
