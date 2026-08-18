#!/bin/bash -l
# Full pre-training leak verification. Three layers:
#   1. _leak_check WITH a trained checkpoint -> end-to-end assertion that the prediction
#      is invariant to the peptide's teacher_bb (the `ckpt` path has never been run).
#   2. target sweep: teacher_atom14/chi/ca/plddt/pae randomised -> prediction must not move.
#   3. mhc_channel_check.py -> does peptide pose leak in through the CO-FOLDED MHC input,
#      and how much accuracy is lost with a foreign same-allele groove (deployment case).
#SBATCH --job-name=mm1_leak
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

CKPT=${CKPT:-checkpoints_mm1/mm1_s1_short/best.pt}
export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || exit 1
mamba activate pmgen2
echo "=== $(date '+%F %T') | LEAK VERIFICATION on $(hostname) | ckpt=$CKPT ==="

echo; echo "###### 1+2. frame leak (end-to-end, trained ckpt) + target sweep ######"
$PY - "$CKPT" <<'PYEOF'
import sys, torch
sys.path.insert(0, "src/model_multimer_1")
import model as MM
MM._leak_check(ckpt=sys.argv[1])
print("PASS: no peptide-frame leak, no target leak (end-to-end, trained weights)")
PYEOF
RC1=$?

echo; echo "###### 3. MHC-conformation channel + deployment case ######"
$PY src/model_multimer_1/mhc_channel_check.py --ckpt "$CKPT" --n 100
RC2=$?

echo; echo "=== $(date '+%F %T') | leakcheck done (frames/targets rc=$RC1, mhc rc=$RC2) ==="
exit $(( RC1 != 0 || RC2 != 0 ))
