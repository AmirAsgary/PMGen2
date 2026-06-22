#!/bin/bash -l
# ============================================================================
# Stratified structural evaluation for model_1 and model_2, plus the groove-
# placement analysis (predicted vs ground-truth peptide->nearest-MHC distance,
# i.e. "does the model adapt to each target or dump every peptide in the canonical
# groove?"). Both are produced by eval_stratified.py; this just drives it for both
# models with sensible defaults.
#
#   bash src/visualization/run_stratified_eval.sh             # run NOW, locally
#   bash src/visualization/run_stratified_eval.sh --submit    # submit to Raven
#
# Override any default via env var, e.g.:
#   MODELS="2" MHC_INIT=truth MAX_GRAPHS=500 \
#       bash src/visualization/run_stratified_eval.sh --submit
# ============================================================================
#SBATCH --job-name=pmgen2_strateval
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval/%j.out
#SBATCH --error=logs/eval/%j.err

set -euo pipefail

# ---- separate the --submit flag from any pass-through args ----
SUBMIT=0
PASS=()
for a in "$@"; do
    if [[ "$a" == "--submit" ]]; then SUBMIT=1; else PASS+=("$a"); fi
done

# ---- if asked to submit and we are NOT already inside a SLURM job, sbatch self ----
if [[ "$SUBMIT" == "1" && -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p logs/eval
    echo "[submit] sbatch $0 ${PASS[*]:-}"
    exec sbatch "$0" ${PASS[@]+"${PASS[@]}"}
fi

# ---- environment: full cluster stack under SLURM, plain python locally ----
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export PYTHONUNBUFFERED=1
    module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
    cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
    mamba activate pmgen2
    PY=$(conda info --base)/envs/pmgen2/bin/python
    mkdir -p logs/eval
    echo "=== $(date '+%F %T') | stratified eval on $(hostname) ==="
    nvidia-smi -L || true
fi
PY=${PY:-python3}                       # local default (override with PY=...)

# ---- configuration (env-overridable) ----
SCHEME=${SCHEME:-two_axis}
FOLD=${FOLD:-1}
MAX_GRAPHS=${MAX_GRAPHS:-1500}          # cap examples for speed (0 = all val)
MODELS=${MODELS:-"1 2"}                 # which models to evaluate
OUT_ROOT=${OUT_ROOT:-outputs/eval_stratified}

CKPT_M1=${CKPT_M1:-checkpoints/two_axis_fold1_variant7_pw5.0_rc3/last.pt}
H5_M1=${H5_M1:-data/processed/h5_store}

CKPT_M2=${CKPT_M2:-checkpoints_model2/mhcdiff_two_axis_fold1/last.pt}
H5_M2=${H5_M2:-data/processed/h5_store_sc}
MHC_INIT=${MHC_INIT:-template}          # template | truth | noise (model_2 only)
N_STEPS=${N_STEPS:-25}                  # model_2 DDIM steps

EVAL=src/post_structure_prediction_processing/eval_stratified.py

echo "[cfg] PY=$PY MODELS='$MODELS' scheme=$SCHEME fold=$FOLD max_graphs=$MAX_GRAPHS"

for m in $MODELS; do
    if [[ "$m" == "1" ]]; then
        echo; echo ">>> model_1  ($CKPT_M1)"
        $PY "$EVAL" --model 1 --ckpt "$CKPT_M1" --h5-dir "$H5_M1" \
            --scheme "$SCHEME" --fold "$FOLD" --max-graphs "$MAX_GRAPHS" \
            --out-dir "$OUT_ROOT/model1" ${PASS[@]+"${PASS[@]}"}
    elif [[ "$m" == "2" ]]; then
        echo; echo ">>> model_2  ($CKPT_M2, mhc-init=$MHC_INIT)"
        $PY "$EVAL" --model 2 --ckpt "$CKPT_M2" --h5-dir "$H5_M2" \
            --scheme "$SCHEME" --fold "$FOLD" --max-graphs "$MAX_GRAPHS" \
            --mhc-init "$MHC_INIT" --n-steps "$N_STEPS" \
            --out-dir "$OUT_ROOT/model2" ${PASS[@]+"${PASS[@]}"}
    else
        echo "[warn] unknown model '$m' (use 1 and/or 2)"
    fi
done

echo; echo "[done] outputs under $OUT_ROOT/{model1,model2}/ "
echo "       stratified.png  +  groove_placement.png  +  per_example.csv"
