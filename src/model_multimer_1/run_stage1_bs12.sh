#!/bin/bash -l
# ============================================================================
# STAGE 1 — LONG RUN at bs=12 (effective 48 across 4 GPUs).
#
# Resumes mm1_s1_final_best_g37400 (gstep 37,400 of the bs=1 run, which had already
# passed every previous run's death point: trig_A 4.5k, trig_C 11.5k, mm1_s1_full 31.9k,
# July 35.75k).
#
# WHY bs=12. Measured (bench_batch.py, 1 GPU, one autograd graph at a time):
#     bs   examples/s   peak GB
#      1        2.23       2.16
#      4        9.85       8.42
#      8       14.53      16.79
#     16       18.43      33.82   <- 89% of 40 GB, too tight
# bs=12 interpolates to ~25 GB typical / ~27 GB worst case. Worst case is only 1.06x
# typical because this dataset is unusually uniform: N mean 192.4, p99 199, MAX 199
# (class I pMHC is ~185 MHC residues + an 8-14mer). No long tail to blow up a batch,
# and near-zero padding waste — which is also why throughput scales so well here.
#
# CORRECTNESS of bs>1 was verified, not assumed: batched loss == mean of individual
# losses to 0.0000 at bs 2/4/8/16. That mattered because openfold's supervised_chi_loss
# computes chi_pi_periodic with einsum("...ij,jk->ik"), which DROPS the batch dim —
# invisible at bs=1, wrong above it. (Our per-example version fixes it.)
#
# LR STAYS AT 3e-4, deliberately NOT scaled up with the batch. Linear scaling would say
# 8x. But a bigger batch REDUCES gradient noise, so 3e-4 at effective-batch-48 is more
# conservative than the 3e-4 at effective-batch-4 that just ran 37k steps clean. LR is
# the knob this model has proven most fragile to (3e-4 and 5e-4 both killed the pre-fix
# architecture). Take the throughput; leave the fragile knob alone.
#
# --epochs 12 => T_max = 12 x 4,268 = 51,216 steps, WHICH IS WHAT THIS JOB RUNS, so the
# cosine anneals to ~0 inside the job. Every long run before today set T_max for 6 epochs
# against a 24h wall clock and truncated at ~15%, making them constant-LR runs in
# disguise; the only run in this project's history that ever survived to completion is
# the one whose LR actually reached zero.
#
# 12 epochs x 204,874 = 2.46M examples in ~11h, vs 410k examples in 13h at bs=1.
# ============================================================================
#SBATCH --job-name=mm1_bs12
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

RESUME=${RESUME:-checkpoints_mm1/mm1_s1_final_best_g37400.pt}
BS=${BS:-12}
EPOCHS=${EPOCHS:-12}
LR=${LR:-3e-4}

$PY - "$RESUME" <<'PYCHK' || exit 1
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["trainable"]
bad = [k for k,v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
if bad: print(f"FATAL: {len(bad)}/{len(sd)} non-finite — refusing."); sys.exit(1)
assert any("attn_norm" in k for k in sd), "FATAL: ckpt lacks the pre-attention LayerNorm"
print(f"  ok: {sys.argv[1]} gstep={ck['global_step']} finite, has pre-attention LayerNorm")
PYCHK

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | STAGE-1 bs=${BS} on $(hostname) | ${NGPU} GPU ==="
echo "    effective batch = ${BS} x ${NGPU} | ${EPOCHS} epochs | lr=${LR} annealing to 0 in-job"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29891 \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir data/processed/h5_store_sc \
    --hasmig-dir data/processed/h5_store_hasmig \
    --data-exp-csv outputs/data_exploration/per_structure.csv \
    --scheme two_axis --fold 1 \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
    --angle-input layernorm --grad-spike-factor 0 \
    --max-val 1500 \
    --ckpt-dir checkpoints_mm1 --run-name mm1_bs12 \
    --resume "$RESUME" --fresh-optim \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --warmup-steps 300 --divergence-factor 8.0 \
    --log-every 50 --train-metrics-every 50 --ckpt-every 500 \
    --num-workers 6 --amp
echo "=== $(date '+%F %T') | bs12 exit $? ==="
