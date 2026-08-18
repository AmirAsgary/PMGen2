#!/bin/bash -l
# ============================================================================
# STAGE 1 — FULL RUN (leak-free, side chains, weighted side-chain losses).
#
# Sizing (measured 2.6 it/s/rank, bs=1, on the short run):
#   train = 204,874 -> 4 GPUs = 51,218 steps/epoch = ~5.5 h/epoch; 6 epochs ~= 33 h.
#   The gpu partition caps at 24 h, so this CANNOT finish in one job. It is designed to
#   be CHAINED: each job runs until the wall clock ends, `last.pt` holds the last
#   COMPLETED epoch, and the next job resumes from it. --epochs must stay 6 in every
#   link so the cosine schedule (t_max = steps/epoch * epochs) stays consistent; the
#   optimizer + scheduler state is restored, so the chain is equivalent to one long run
#   apart from losing the partial epoch at each cut (~2 h).
#
#     sbatch src/model_multimer_1/run_stage1_full.sh                    # link 1
#     sbatch --dependency=afterany:<jobid> RESUME=auto \
#            src/model_multimer_1/run_stage1_full.sh                    # link 2, etc.
#   (RESUME=auto picks checkpoints_mm1/$RUN/last.pt if it exists.)
#
# --max-val 3000: the full val fold is 30,239 structures = ~42 min per pass, x2 for
# val-matched, EVERY epoch — ~8 h over the run, more than an epoch of training. 3000 is
# seeded, so it is the SAME subset every epoch and across links, and stays comparable
# to the short run's numbers.
#
# Side-chain losses are WEIGHTED by default now (peptide/sample/struct actually apply);
# --unweighted-sidechain-losses restores the old behaviour if you want the A/B.
# ============================================================================
#SBATCH --job-name=mm1_s1_full
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=23:30:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

H5=${H5:-data/processed/h5_store_sc}
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
RUN=${RUN:-mm1_s1_full}
EPOCHS=${EPOCHS:-6}
LR=${LR:-5e-4}
MAX_VAL=${MAX_VAL:-3000}
RESUME=${RESUME:-}
# AF2's probabilistic FAPE clamp. OFF from scratch on purpose: at init the peptide is
# >10 A off, so a 10 A clamp zeroes the gradient and FAPE stalls (measured: 0.92, pep-RMSD
# 10.8 A). That argument is about EARLY training only. On a RESUME from a converged
# checkpoint (~1.4 A) the clamp essentially never binds on normal examples and bounds
# only the heavy-tailed outliers that produce the gradient spikes which killed job
# 29377637. 10 A / p=0.9 is AlphaFold's own setting.
FAPE_CLAMP=${FAPE_CLAMP:-0}
FAPE_CLAMP_PROB=${FAPE_CLAMP_PROB:-0.9}
GRAD_SPIKE=${GRAD_SPIKE:-10.0}

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing"; exit 1; }
done

# RESUME=auto -> continue this run's own last.pt if present
# RESUME=auto -> last.pt (the furthest-along epoch) for a NORMAL wall-clock chain.
# WARNING: after a DIVERGENCE ABORT, last.pt is the collapsed state (it is written every
# --ckpt-every steps and is finite, just bad) while best.pt holds the last good weights.
# Job 29377637: last.pt total 2.24 / Ca-RMSD 17.7 A vs best.pt total 0.227 / pep-RMSD 1.4 A.
# After any failed run, resume from best.pt EXPLICITLY.
if [[ "$RESUME" == "auto" ]]; then
    RESUME="checkpoints_mm1/${RUN}/last.pt"
    [[ -f "$RESUME" ]] || { echo "[mm1] RESUME=auto but no $RESUME yet — starting fresh"; RESUME=""; }
fi
RESUME_FLAG=""
if [[ -n "$RESUME" ]]; then
    # refuse to resume from a poisoned checkpoint
    $PY - "$RESUME" <<'PYCHK' || exit 1
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("trainable", {})
bad = [k for k, v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
if bad:
    print(f"FATAL: {sys.argv[1]} has {len(bad)}/{len(sd)} non-finite tensors — refusing to resume.")
    sys.exit(1)
print(f"  ok: {sys.argv[1]} is finite ({len(sd)} tensors)")
PYCHK
    RESUME_FLAG="--resume $RESUME"
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU(s) | lr=${LR} ==="
echo "    stage 1 | pep-frames=identity (leak-free) | sidechains ON, WEIGHTED"
echo "    epochs=${EPOCHS} max-val=${MAX_VAL} ${RESUME_FLAG}"
echo "    guards: grad-spike-factor=${GRAD_SPIKE} fape-clamp=${FAPE_CLAMP} (p=${FAPE_CLAMP_PROB})"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29821 \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme two_axis --fold 1 \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
    --grad-spike-factor ${GRAD_SPIKE} --grad-spike-warmup 200 \
    --fape-clamp ${FAPE_CLAMP} --fape-clamp-prob ${FAPE_CLAMP_PROB} \
    --max-val ${MAX_VAL} \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" $RESUME_FLAG \
    --epochs ${EPOCHS} --bs 1 --lr ${LR} \
    --warmup-steps 1000 --divergence-factor 8.0 \
    --log-every 100 --train-metrics-every 100 --ckpt-every 1000 \
    --num-workers 6 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
