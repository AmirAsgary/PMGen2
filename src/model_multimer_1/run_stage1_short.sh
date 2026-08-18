#!/bin/bash -l
# ============================================================================
# STAGE 1 — SHORT HARNESS RUN.  Purpose: get the FIRST honest held-out number.
#
# Context: the only leak-free run ever attempted (job 28811055, 13 Jul) diverged at
# ~24.8k steps, sat dead for 8k steps, went NaN, and was cancelled at 35% of epoch 1
# of 6. It therefore NEVER reached a validation pass — every held-out number quoted
# for this model comes either from the leaky checkpoints or from 20-40 structure
# probes. Its last.pt is 292/292 NaN tensors. There is no usable checkpoint.
#
# The stability guards written afterwards (non-finite step skip, NaN-safe save,
# best.pt, divergence abort, |g| telemetry, LR warmup, --trunk-fp32) have never been
# exercised by a real job. This run exercises them, and the validation path, cheaply.
#
# What is NEW here versus run_stage1_sidechain.sh:
#   * sc-FAPE and chi are now WEIGHTED (peptide_weight / sample_weight /
#     struct_weight actually reach them). They are ~71% of the total loss and were
#     ~95% MHC, i.e. mostly rescoring rotamers of a backbone handed in as input.
#     --unweighted-sidechain-losses restores the old behaviour to A/B against.
#   * --cap-train, NOT --max-train: --max-train replaces val with a random slice of
#     the TRAIN pool (shares alleles+peptides with train -> optimistic). --cap-train
#     keeps the real two_axis held-out HLA fold, so the val number means something.
#   * --max-val: the full val fold is ~30k structures and would cost more than the
#     training it follows. 2000 seeded structures, identical every epoch.
#   * evaluate() now runs under no_grad (it did not before).
#
# NOT a converged model — 6k structures x 3 epochs is ~9k optimizer steps per rank,
# well short of the ~24.8k where the old run blew up. It answers: does the pipeline
# train, validate, and checkpoint honestly end-to-end? Launch the long run after.
#
#   sbatch src/model_multimer_1/run_stage1_short.sh
# ============================================================================
#SBATCH --job-name=mm1_s1_short
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

H5=${H5:-data/processed/h5_store_sc}
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
RUN=${RUN:-mm1_s1_short}
CAP_TRAIN=${CAP_TRAIN:-6000}
MAX_VAL=${MAX_VAL:-1500}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-4}

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

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU(s) ==="
echo "    cap-train=${CAP_TRAIN} max-val=${MAX_VAL} epochs=${EPOCHS} lr=${LR}"
echo "    sidechain losses WEIGHTED (peptide/sample/struct now apply)"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29814 \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme two_axis --fold 1 \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 --trunk-fp32 tri \
    --cap-train ${CAP_TRAIN} --max-val ${MAX_VAL} \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" \
    --epochs ${EPOCHS} --bs 1 --lr ${LR} \
    --warmup-steps 500 --divergence-factor 8.0 \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 \
    --num-workers 6 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
