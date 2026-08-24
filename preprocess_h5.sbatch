#!/bin/bash
#SBATCH --job-name=prep_%a
#SBATCH --array=1-222                 # number of chunks
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/prep.%A_%a.out
#SBATCH -e ./logs/prep.%A_%a.err
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2"
mamba activate pmgen2
$PY src/model/preprocess.py \
    --chunks-dir "$HOME/projects/PMGen_2/data/pmgen_inputs/chunks" \
    --out-dir    data/processed/h5_store \
    --chunks "chunk_${SLURM_ARRAY_TASK_ID}" \
    --no-merge                        # merge once at the end (next line)

# after the array finishes, merge per-chunk indices into index.csv:
