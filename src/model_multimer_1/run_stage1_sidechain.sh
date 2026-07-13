#!/bin/bash -l
# ============================================================================
# STAGE 1 RETRAIN — leak-free + AlphaFold-style side chains. From SCRATCH.
#
# Why from scratch: every earlier checkpoint was trained with the peptide-pose LEAK
# (`_frames_from_bb` fed the peptide's true backbone frames to the trunk), so its trunk
# learned to read the answer. Those weights are not salvageable.
#
# What this run does differently:
#   --pep-frames identity  peptide frames withheld -> the pose must be PREDICTED.
#                          train.py runs `_leak_check` pre-flight and aborts on a leak.
#   --sidechains           AF2 side-chain supervision: sidechain-FAPE (0.5) over the 8
#                          rigid-group frames + atom14 with symmetric-atom renaming, and
#                          supervised_chi (1.0; chi 0.5, angle_norm 0.01). Backbone FAPE
#                          0.5. The StructureModule stays 100% FROZEN; a trainable
#                          AngleResnet predicts the torsions.
#   stage 1                structure ONLY: lambda_plddt = 0 and plddt_proj FROZEN.
#   lr 5e-4                2e-3 diverges to NaN by epoch 15; 5e-4 is stable.
#
# Data (verified): TRAIN = old-store two_axis TRAIN ids (confidence-filtered) + ALL
# hasmig ids (sample_weight 1.0 in stage 1). VAL = old-store two_axis VAL ids ONLY —
# hasmig is never validated on. train/val are disjoint and no val/test base id ever
# appears in train.
#
# PREREQUISITE — both stores must carry the side-chain targets, or this job aborts:
#   sbatch src/model_multimer_1/preprocess_hasmig.sbatch      # hasmig, --sidechains
#   python src/model_multimer_1/preprocess_hasmig.py --merge --out-dir <hasmig> --csv x --zip-dir x
#   sbatch src/model/reprocess_sidechains.sbatch              # old store, --sidechains
#   python src/model_multimer_1/model.py                      # smoke + leak-check
#
#   sbatch src/model_multimer_1/run_stage1_sidechain.sh
# ============================================================================
#SBATCH --job-name=mm1_s1_sc
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mm1/%j.out
#SBATCH --error=logs/mm1/%j.err

H5=${H5:-data/processed/h5_store_sc}
HASMIG=${HASMIG:-data/processed/h5_store_hasmig}
DATAEXP=${DATAEXP:-outputs/data_exploration/per_structure.csv}
SCHEME=two_axis
FOLD=1
RUN=mm1_stage1_sc
EPOCHS=6
BS=1
LR=5e-4                 # 2e-3 -> NaN by epoch 15; 5e-4 verified stable
RESUME=${RESUME:-}      # set to checkpoints_mm1/mm1_stage1_sc/last.pt to continue

export PYTHONUNBUFFERED=1
module load apptainer gcc/13 cuda/12.6 openmpi_gpu/5.0
PY=$(conda info --base)/envs/pmgen2/bin/python
cd "$HOME/projects/PMGen_2/PMGen2" || { echo "FATAL: repo dir missing"; exit 1; }
mamba activate pmgen2
mkdir -p logs/mm1

for f in params/alphafold/input_embedder_mm.pt params/alphafold/sm_mm.pt \
         params/alphafold/plddt_mm.pt; do
    [[ -f "$f" ]] || { echo "FATAL: $f missing (download+extract weights first)"; exit 1; }
done

# fail fast if a store lacks the side-chain targets (train.py re-checks, but this gives a
# clear message before the 4-GPU allocation is spent)
$PY - "$H5" "$HASMIG" <<'PYCHK' || exit 1
import glob, sys, h5py
for d in sys.argv[1:]:
    sh = sorted(glob.glob(f"{d}/*.h5"))
    if not sh:
        print(f"FATAL: no shards in {d}"); sys.exit(1)
    with h5py.File(sh[0]) as h:
        g = h[next(iter(h.keys()))]
        miss = [k for k in ("teacher_atom14", "teacher_chi") if k not in g]
        if miss:
            print(f"FATAL: {d} lacks {miss} -- re-run preprocessing with --sidechains")
            sys.exit(1)
    print(f"  ok: {d} has side-chain targets")
PYCHK

RESUME_FLAG=""
[[ -n "$RESUME" ]] && RESUME_FLAG="--resume $RESUME"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== $(date '+%F %T') | ${RUN} on $(hostname) | ${NGPU} GPU(s) | lr=${LR} ==="
echo "    pep-frames=identity (leak-free)  sidechains=ON  stage=1 (plddt weight 0)"

$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=${NGPU} \
    --master_addr=127.0.0.1 --master_port=29811 \
    src/model_multimer_1/train.py \
    --stage 1 --pep-frames identity --sidechains \
    --h5-dir "$H5" --hasmig-dir "$HASMIG" --data-exp-csv "$DATAEXP" \
    --scheme "$SCHEME" --fold "$FOLD" \
    --bb-fape-w 0.5 --sc-fape-w 0.5 --chi-w 1.0 \
    --n-trunk 3 --mhc-noise 0.1 --grad-clip 1.0 \
    --ckpt-dir checkpoints_mm1 --run-name "$RUN" $RESUME_FLAG \
    --epochs ${EPOCHS} --bs ${BS} --lr ${LR} \
    --log-every 50 --train-metrics-every 50 --ckpt-every 1000 \
    --num-workers 4 --amp

RC=$?
echo "=== $(date '+%F %T') | ${RUN} exit ${RC} ==="
exit $RC
