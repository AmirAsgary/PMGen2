#!/bin/bash -l
# ============================================================================
# CONTINUE Approach A from its CURRENT checkpoint, into a NEW run dir.
#   reads : checkpoints_mm1/mm1_stage2_A/last.pt        (never overwritten)
#   writes: checkpoints_mm1/mm1_stage2_A_resume/        (new)
# --fresh-optim => weights are kept but the optimizer + LR schedule restart (the
# original A finished with its cosine LR annealed to 0, so a plain resume would not
# train). Same data/loss as A: filtered train + filtered val.
#
#   sbatch src/model_multimer_1/run_A_resume.sh
# (edit EPOCHS/LR below to taste; they are hardcoded, not env vars, on purpose —
#  SLURM does not reliably propagate submit-time env vars into the job.)
# ============================================================================
#SBATCH --job-name=mm1_s2A_res
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

RESUME=checkpoints_mm1/mm1_stage2_A/last.pt
RUN=mm1_stage2_A_resume
EPOCHS=6
LR=1e-4

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

[[ -f "$RESUME" ]] || { echo "FATAL: $RESUME missing (need A's current checkpoint)"; exit 1; }
for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing"; exit 1; }
done

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} (resume A) on $(hostname) | ${NGPU} GPU(s) | epochs=${EPOCHS} lr=${LR} ==="
echo "    resume=${RESUME} (read-only) -> checkpoints_mm1/${RUN}"
$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29811 \
    src/model_multimer_1/train.py \
    --stage 2 --resume "$RESUME" --fresh-optim \
    --h5-dir data/processed/h5_store --hasmig-dir data/processed/h5_store_hasmig \
    --data-exp-csv outputs/data_exploration/per_structure.csv \
    --scheme two_axis --fold 1 \
    --force-filter --filter-val \
    --hasmig-weight 0.1 --plddt-w 0.01 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --epochs ${EPOCHS} --bs 1 --lr ${LR} \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
