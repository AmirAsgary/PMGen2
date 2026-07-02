#!/bin/bash -l
# ============================================================================
# Train model_multimer_1. Three stages (set STAGE):
#   STAGE=1 : high-confidence + high-burial subset, trunk+SM trainable (pLDDT head
#             frozen). Structure-first: pLDDT weight 0 (epoch 1) then --plddt-w (0.01).
#   STAGE=2 : ALL structures, SM stays trainable, pLDDT weight 0.01 (structure+conf).
#             point RESUME at stage-1's last.pt to carry the structure over.
#   STAGE=3 : freeze the WHOLE model except the pLDDT projection (pLDDT head always
#             frozen); confidence-only, pLDDT weight 1.0. RESUME from stage-2.
#
#   STAGE=1 sbatch src/model_multimer_1/train_mm1.sbatch
#   STAGE=2 RESUME=checkpoints_mm1/mm1_stage1/last.pt sbatch src/model_multimer_1/train_mm1.sbatch
#   STAGE=3 RESUME=checkpoints_mm1/mm1_stage2/last.pt sbatch src/model_multimer_1/train_mm1.sbatch
# ============================================================================
#SBATCH --job-name=mm1
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

STAGE=${STAGE:-1}
H5=${H5:-data/processed/h5_store}
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
SCHEME=${SCHEME:-two_axis}                 # old-store HLA split for val/test
FOLD=${FOLD:-1}                            # CV fold (1..5)
EPOCHS=${EPOCHS:-8}
BS=${BS:-1}
LR=${LR:-1e-4}
RESUME=${RESUME:-}
RUN="mm1_stage${STAGE}"

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
if [[ "$STAGE" == "2" && -z "$RESUME" && -f "checkpoints_mm1/mm1_stage1/last.pt" ]]; then
    RESUME_FLAG="--resume checkpoints_mm1/mm1_stage1/last.pt"
fi
if [[ "$STAGE" == "3" && -z "$RESUME" && -f "checkpoints_mm1/mm1_stage2/last.pt" ]]; then
    RESUME_FLAG="--resume checkpoints_mm1/mm1_stage2/last.pt"
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
PORT=$((29700 + STAGE))
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU(s) ==="
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=${PORT} \
    src/model_multimer_1/train.py \
    --stage "$STAGE" --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme "$SCHEME" --fold "$FOLD" \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" $RESUME_FLAG \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --ckpt-every 1000 --num-workers 4 --amp

exit $?
