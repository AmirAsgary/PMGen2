#!/bin/bash -l
# ============================================================================
# STAGE 2 from the preserved stage-1 checkpoint — two alternative approaches,
# meant to be submitted IN PARALLEL (set APPROACH=A or B):
#
#   A  continue on FILTERED structures; validate on the FILTERED val subset only.
#      (--force-filter --filter-val)                        -> run mm1_stage2_A
#   B  FULL dataset, no filter; weight the STRUCTURAL (FAPE) loss per structure by
#      w_n = w_min + (1-w_min)·q_plddt·q_burial (low-quality structures still
#      contribute weakly instead of being dropped); validate on ALL structures.
#      (--struct-quality-weight)                            -> run mm1_stage2_B
#
# Both: SM FROZEN (as always), pLDDT weight 0.01, hasmig down-weighted 0.1, 2 epochs,
# resumed from a FROZEN copy of stage-1's LAST checkpoint (so A and B share the exact
# same initialization). The copy is made once, idempotently.
#
# Pass the approach as the FIRST ARGUMENT (a positional arg SLURM always forwards to the
# script as $1 — unlike `APPROACH=B sbatch ...`, whose env var is NOT reliably propagated
# to the batch job on every cluster):
#
#   sbatch src/model_multimer_1/stage2_mm1.sbatch A
#   sbatch src/model_multimer_1/stage2_mm1.sbatch B
#
# Two parallel jobs at gpu:a100:4 need 8 GPUs; if your quota is 4, lower --gres to
# gpu:a100:2 here (2 epochs still fit in 24 h) or submit B after A.
# ============================================================================
#SBATCH --job-name=mm1_s2
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

# approach from the FIRST POSITIONAL ARG (preferred), else the APPROACH env var; no
# silent default — a missing/typo'd approach FAILS instead of quietly running A.
APPROACH=${1:-${APPROACH:-}}
H5=${H5:-data/processed/h5_store}
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
SCHEME=${SCHEME:-two_axis}
FOLD=${FOLD:-1}
EPOCHS=${EPOCHS:-2}
BS=${BS:-1}
LR=${LR:-1e-4}
PLDDT_W=${PLDDT_W:-0.01}
HASMIG_W=${HASMIG_W:-0.1}
WMIN=${WMIN:-0.05}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

# --- preserve stage-1's LAST checkpoint as a frozen pretrained snapshot (once) ------
PRETRAINED=checkpoints_mm1/pretrained_stage1_last.pt
SRC=checkpoints_mm1/mm1_stage1/last.pt
if [[ ! -f "$PRETRAINED" ]]; then
    [[ -f "$SRC" ]] || { echo "FATAL: $SRC missing (stage-1 checkpoint)"; exit 1; }
    cp "$SRC" "${PRETRAINED}.tmp.$$" && mv -n "${PRETRAINED}.tmp.$$" "$PRETRAINED"
    rm -f "${PRETRAINED}.tmp.$$"
    echo "preserved stage-1 LAST checkpoint -> $PRETRAINED"
fi

case "$APPROACH" in
  A) FLAGS="--force-filter --filter-val";        RUN=mm1_stage2_A; PORT=29801 ;;
  B) FLAGS="--struct-quality-weight --w-min $WMIN"; RUN=mm1_stage2_B; PORT=29802 ;;
  *) echo "FATAL: approach must be A or B (got '${APPROACH}'). Pass it as the first"\
          "argument, e.g.: sbatch src/model_multimer_1/stage2_mm1.sbatch B"; exit 1 ;;
esac

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing"; exit 1; }
done

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} (Approach ${APPROACH}) on $(hostname) | ${NGPU} GPU(s) ==="
echo "    resume=${PRETRAINED} flags='${FLAGS}' plddt_w=${PLDDT_W} hasmig_w=${HASMIG_W}"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model_multimer_1/train.py \
    --stage 2 --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme "$SCHEME" --fold "$FOLD" --resume "$PRETRAINED" \
    --plddt-w "$PLDDT_W" --hasmig-weight "$HASMIG_W" $FLAGS \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --ckpt-every 1000 --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
