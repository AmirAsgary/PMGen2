#!/bin/bash -l
#SBATCH --job-name=mm1_bench
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
mkdir -p logs/mm1

CKPT=${CKPT:-checkpoints_mm1/ARCHIVE/stage2b_best.pt}

echo "########## MHC input = the target's own crystal groove (upper bound) ##########"
$PY benchmark/run_benchmark.py --ckpt "$CKPT" --mhc-source crystal --leak-test

echo
echo "########## MHC input = a FOREIGN groove from another complex (deployment) ##########"
$PY benchmark/run_benchmark.py --ckpt "$CKPT" --mhc-source template
