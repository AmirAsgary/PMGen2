#!/bin/bash -l
# ============================================================================
# STAGE 2 — add the confidence term, keep the structure objective intact.
#
# Resumes the archived stage-1 best (val-matched 1.34 A, test-set peptide CA 1.851 A
# mean / 0.962 A median, full-atom 2.674 A mean).
#
# WHAT CHANGES vs stage 1 — exactly one thing: lambda_plddt 0 -> 0.01.
#
#   --plddt-w 0.01        measured, not inherited: on real batches the pLDDT CE gradient
#                         is 15.5x the structural one at lambda=1, so 0.01 puts confidence
#                         at ~15.5% of the structural gradient. Balanced.
#   --hasmig-weight 1.0   stage 2 DEFAULTS to 0.1; kept at 1.0 so hasmig is not
#                         down-weighted. hasmig is 130,993 structures at median pep-pLDDT
#                         83.4 with ONE structure per complex — the cleanest data in the
#                         project. Down-weighting it was never justified by a measurement.
#   NO --force-filter     stage 2 drops the confidence filter, and here that is the POINT.
#                         Stage 1 trains on burial>=0.65 AND pep-pLDDT>70 = ~8.8% of the
#                         data, i.e. ONLY high-confidence examples. A model that never
#                         sees the low-confidence tail cannot learn to REPORT low
#                         confidence — and non-discriminative pLDDT is precisely the gap
#                         (test set: predicted 72.8-79.6 while true error spans 0.5-5.1 A).
#   --struct-quality-weight
#                         with the filter gone, low-quality structures would otherwise
#                         pollute FAPE. This scales the STRUCTURAL loss per structure by
#                         w_n = w_min + (1-w_min)*q_plddt*q_burial, leaving the pLDDT CE
#                         at FULL UNIFORM weight on every structure. So the structure
#                         objective stays effectively as it was (bad structures contribute
#                         ~w_min=0.05, near-zero, which is what filtering did), while the
#                         confidence term sees the whole confidence range at a flat 0.01,
#                         independent of quality and burial.
#                         NB this only works because of the fix in 65dfde2: struct_weight
#                         previously never reached sc_fape/chi (71% of the loss), so this
#                         flag was inert before.
#
# EPOCH SIZING — recompute it, do not copy stage 1's. Dropping the filter takes the train
# set from 204,874 to 967,149, so an epoch is 20,149 steps instead of 4,269 (4.7x). At
# ~1.4 it/s that is ~4 h/epoch, so --epochs 5 => T_max = 100,745 steps ~= 20 h, and the
# cosine ANNEALS TO ~0 INSIDE THE JOB. (--epochs 16 would have been 64 h against a 23.5 h
# cap and truncated the schedule at ~36% — the exact failure mode that made every long run
# before 2026-08-20 a constant-LR run in disguise.)
#
# Structure-loss weights, architecture, LR and schedule are otherwise identical to the
# stage-1 run, so any change in the structural metrics is attributable to the data/weight
# change rather than to a different objective.
# ============================================================================
#SBATCH --job-name=mm1_stage2
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=23:30:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
mkdir -p logs/mm1

RUN=${RUN:-mm1_stage2}
EPOCHS=${EPOCHS:-5}
LR=${LR:-2e-4}
SEED_CKPT=${RESUME:-checkpoints_mm1/ARCHIVE/stage1_best_20260821.pt}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}

for BS in 12 10 8; do
    CKPT="$SEED_CKPT"
    [[ -f "checkpoints_mm1/${RUN}/last.pt" ]] && CKPT="checkpoints_mm1/${RUN}/last.pt"
    $PY - "$CKPT" <<'PYCHK' || exit 1
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["trainable"]
bad = [k for k,v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
if bad: print(f"FATAL: {len(bad)}/{len(sd)} non-finite — refusing."); sys.exit(1)
assert any("attn_norm" in k for k in sd), "FATAL: ckpt lacks the pre-attention LayerNorm"
print(f"  ok: {sys.argv[1]} gstep={ck['global_step']} stage={ck['stage']} finite, LayerNorm present")
PYCHK
    MARK="logs/mm1/${SLURM_JOB_ID}.bs${BS}.err"
    echo "=== $(date '+%F %T') | ${RUN} attempt bs=${BS} | resume=${CKPT} ==="
    $PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
        --master_addr=127.0.0.1 --master_port=29911 \
        src/model_multimer_1/train.py \
        --stage 2 --pep-frames identity --sidechains \
        --h5-dir data/processed/h5_store_sc \
        --hasmig-dir data/processed/h5_store_hasmig \
        --data-exp-csv outputs/data_exploration/per_structure.csv \
        --scheme two_axis --fold 1 \
        --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
        --plddt-w 0.01 --hasmig-weight 1.0 --struct-quality-weight \
        --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
        --angle-input layernorm --grad-spike-factor 0 \
        --max-val 1500 \
        --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
        --resume "$CKPT" --fresh-optim \
        --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
        --warmup-steps 300 --divergence-factor 8.0 \
        --log-every 50 --train-metrics-every 50 --ckpt-every 500 \
        --num-workers 6 --amp 2> >(tee "$MARK" >&2)
    RC=$?
    echo "=== $(date '+%F %T') | ${RUN} bs=${BS} exit ${RC} ==="
    [[ $RC -eq 0 ]] && { echo "COMPLETED at bs=${BS}"; break; }
    if grep -qiE "out of memory|OutOfMemoryError|CUDA out of memory" "$MARK" 2>/dev/null; then
        echo ">>> OOM at bs=${BS}; falling back, resuming from this run's own last.pt"; continue
    fi
    echo ">>> non-OOM failure (rc=${RC}); not retrying at a smaller batch."; break
done
echo "=== $(date '+%F %T') | ${RUN} finished ==="
