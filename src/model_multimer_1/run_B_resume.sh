#!/bin/bash -l
# ============================================================================
# CONTINUE Approach B from its CURRENT checkpoint, into a NEW run dir.
#   reads : checkpoints_mm1/mm1_stage2_B/last.pt        (never overwritten)
#   writes: checkpoints_mm1/mm1_stage2_B_resume/        (new)
# --fresh-optim => weights kept, optimizer + LR schedule restart (B was cut off by
# the 1-day limit mid-epoch-1, so a fresh, longer schedule lets it keep learning).
# Same data/loss as B: FULL dataset (no filter), FAPE quality-weighted by w_n.
#
#   sbatch src/model_multimer_1/run_B_resume.sh
# (edit EPOCHS/LR below to taste; hardcoded, not env vars, on purpose.)
# ============================================================================
#SBATCH --job-name=mm1_s2B_res
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

RESUME=checkpoints_mm1/mm1_stage2_B/last.pt
RUN=mm1_stage2_B_resume
EPOCHS=6
LR=2e-4

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

[[ -f "$RESUME" ]] || { echo "FATAL: $RESUME missing (need B's current checkpoint)"; exit 1; }
for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing"; exit 1; }
done

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} (resume B) on $(hostname) | ${NGPU} GPU(s) | epochs=${EPOCHS} lr=${LR} ==="
echo "    resume=${RESUME} (read-only) -> checkpoints_mm1/${RUN}"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29812 \
    src/model_multimer_1/train.py \
    --stage 2 --resume "$RESUME" --fresh-optim \
    --h5-dir data/processed/h5_store --hasmig-dir data/processed/h5_store_hasmig \
    --data-exp-csv outputs/data_exploration/per_structure.csv \
    --scheme two_axis --fold 1 \
    --struct-quality-weight --w-min 0.05 \
    --hasmig-weight 0.1 --plddt-w 0.01 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --epochs ${EPOCHS} --bs 1 --lr ${LR} \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
