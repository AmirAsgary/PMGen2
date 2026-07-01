#!/bin/bash -l
# ============================================================================
# One-pass data exploration over the teacher H5 store: peptide pLDDT + the
# MHC-coordination burial score. Writes plots + per_structure.csv.
# CPU-only work, but we request a (mostly idle) GPU so SLURM routes it through the
# gpu allocation and avoids the small-QOS CPU cap.
#
#   sbatch src/visualization/data_exploration.sbatch
#   H5=data/processed/h5_store_sc sbatch src/visualization/data_exploration.sbatch
# ============================================================================
#SBATCH --job-name=pmgen2_dataexp
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dataexp_%j.out
#SBATCH --error=logs/dataexp_%j.err

H5=${H5:-data/processed/h5_store}
OUT=${OUT:-outputs/data_exploration}
RADIUS=${RADIUS:-10}
C0=${C0:-6}
SC=${SC:-2}
MAXN=${MAXN:-0}                     # 0 = all structures; set e.g. 5000 for a quick test

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2

mkdir -p logs "$OUT"
[[ -f "$H5/index.csv" ]] || { echo "FATAL: $H5/index.csv not found"; exit 1; }
echo "=== $(date '+%F %T') | data_exploration on $(hostname) | H5=$H5 ==="

$PY src/visualization/data_exploration.py \
    --h5-dir "$H5" --out-dir "$OUT" \
    --radius "$RADIUS" --c0 "$C0" --s-c "$SC" --max "$MAXN"

exit $?
