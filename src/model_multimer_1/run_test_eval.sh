#!/bin/bash -l
# Evaluate a checkpoint on the 15 class-I examples in data/test/pdbs/alphafold/
# (3 complexes x ~6 ANCHOR VARIANTS each, with AlphaFold's structure as reference).
# Writes full-atom PDBs (chain A = MHC, chain B = peptide, b-factor = predicted pLDDT),
# reports Ca / backbone / FULL-ATOM peptide RMSD after MHC-Ca superposition, and runs
# the leakage probe on these exact structures.
#SBATCH --job-name=mm1_test
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

# POSITIONAL args, not env vars: SLURM does not reliably propagate `VAR=x sbatch ...`
# to the batch job (the same trap the stage-2 A/B launcher documents). If it silently
# falls back to the default you evaluate the WRONG checkpoint and never notice, because
# the run still succeeds.
#   sbatch run_test_eval.sh <ckpt> <tag> [outdir]
CKPT=${1:-${CKPT:-checkpoints_mm1/snapshots/s1_full_mid.pt}}
TAG=${2:-${TAG:-s1full_mid}}
OUT=${3:-${OUT:-outputs/mm1_test}}
export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
echo "=== $(date '+%F %T') | test eval on $(hostname) ==="
echo "    CKPT=$CKPT"
echo "    TAG=$TAG  OUT=$OUT"

echo; echo "########## LEAK-FREE (pep-frames identity) + leak probe ##########"
$PY src/model_multimer_1/predict_test.py --ckpt "$CKPT" --tag "$TAG" \
    --pep-frames identity --stage 1 --out-dir "$OUT" --leak-test
RC=$?

echo; echo "########## CONTROL: same ckpt with the LEAKY frames, for reference ##########"
echo "(this checkpoint was TRAINED with identity frames, so 'teacher' here is an"
echo " out-of-distribution input, not a valid model — it is only a sanity contrast.)"
$PY src/model_multimer_1/predict_test.py --ckpt "$CKPT" --tag "$TAG" \
    --pep-frames teacher --stage 1 --out-dir "$OUT" --leak-test

echo; echo "=== $(date '+%F %T') | test eval done (rc=$RC) ==="
exit $RC
