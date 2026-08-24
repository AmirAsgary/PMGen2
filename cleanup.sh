#!/bin/bash
# ============================================================================
# PMGen2 repository cleanup.
#
# DRY RUN BY DEFAULT.  Nothing is deleted unless you pass --apply, and with
# --apply every group still asks before it touches anything.
#
#   ./cleanup.sh                      # show what would go, delete nothing
#   ./cleanup.sh --apply              # ask per group, then delete
#   ./cleanup.sh --apply --only h5-backups
#   ./cleanup.sh --list               # just the group names
#
# Three things this script will not do:
#   * touch anything under the flow-matching model (another agent owns it),
#   * touch a file the mm1 pipeline or the benchmark still reads,
#   * delete the H5 backup stores without re-proving, at run time, that the
#     live store is a strict superset of them.
# Every deletion is checked against PROTECTED[] immediately before it happens.
# ============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1
ROOT="$PWD"
PY="$(conda info --base 2>/dev/null)/envs/pmgen2/bin/python"
[[ -x "$PY" ]] || PY=python3

APPLY=0; ASSUME_YES=0; ONLY=""; SELFTEST=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --only) shift ;;
    --only=*) ONLY="${a#--only=}" ;;
    --list) echo "h5-backups hasmig-tar afdb old-checkpoints mm1-probes dead-code old-logs pycache"; exit 0 ;;
    --self-test) SELFTEST=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) [[ -n "${PREV:-}" ]] && ONLY="$a"; ;;
  esac
  PREV="$a"
done

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'

# ---------------------------------------------------------------------------
# Paths that must survive, no matter what a group asks for.
# The flow-matching model belongs to a parallel effort; utils.py and parse.py
# are imported by BOTH models; per_structure.csv is --data-exp-csv for mm1
# training and is read by six flow-model scripts; ARCHIVE holds every named
# checkpoint including the one the benchmark defaults to.
# ---------------------------------------------------------------------------
PROTECTED=(
  "src/flow_model" "checkpoints_flow" "logs/flow"
  "data/processed/flow_canonical" "data/processed/flow_canonical_smoke"
  "src/model/utils.py" "src/pdb" "src/model/preprocess.py"
  "src/model_multimer_1/model.py" "src/model_multimer_1/train.py"
  "src/model_multimer_1/utils.py" "src/model_multimer_1/predict_test.py"
  "src/model_multimer_1/preprocess_hasmig.py"
  "outputs/data_exploration" "outputs/mm1_test" "outputs/mm1_anchor"
  "params" "openfold" "benchmark" "data/test" "data/processed/two_axis"
  "data/processed/h5_store_sc" "data/processed/h5_store_hasmig"
  "checkpoints_mm1/ARCHIVE" ".git"
)

TOTAL_KB=0
declare -a PENDING

guard() {
  local p="$1" abs
  [[ -z "$p" || "$p" == "/" || "$p" == "." || "$p" == ".." ]] && {
      echo "${RED}REFUSING: degenerate path '$p'${OFF}"; return 1; }
  abs="$(readlink -f "$p" 2>/dev/null)"
  [[ -z "$abs" ]] && return 1
  case "$abs" in
    "$ROOT"/*) ;;
    *) echo "${RED}REFUSING: '$p' resolves outside the repo ($abs)${OFF}"; return 1 ;;
  esac
  local prot pabs
  for prot in "${PROTECTED[@]}"; do
    pabs="$(readlink -f "$ROOT/$prot" 2>/dev/null)" || continue
    [[ -z "$pabs" ]] && continue
    # refuse if the target IS a protected path or CONTAINS one
    if [[ "$abs" == "$pabs" || "$abs" == "$pabs"/* || "$pabs" == "$abs"/* ]]; then
      echo "${RED}REFUSING: '$p' collides with protected '$prot'${OFF}"; return 1
    fi
  done
  return 0
}

if (( SELFTEST )); then
  # The guard is the only thing standing between a typo and 500 GB, so prove it
  # refuses what it must and permits what the groups actually need to remove.
  fails=0
  must_refuse=(src/flow_model checkpoints_flow logs/flow data/processed/flow_canonical
               src/model/utils.py src/pdb/parse.py src/pdb src/model/preprocess.py
               src/model_multimer_1/model.py src/model_multimer_1/train.py
               outputs/data_exploration outputs/data_exploration/per_structure.csv
               params openfold benchmark data/test data/processed/two_axis
               data/processed/h5_store_sc data/processed/h5_store_hasmig
               checkpoints_mm1/ARCHIVE checkpoints_mm1/ARCHIVE/stage2b_best.pt
               .git src "" / .. ../PMGen2 /etc /u/amirasgary)
  must_allow=(src/model_2 src/model_3 src/model4 logs/distill logs/diff tmp tmp.txt
              checkpoints checkpoints_model2 checkpoints_mm1_bk
              checkpoints_mm1/mm1_s1_full checkpoints_mm1/trig_A
              data/afdb data/processed/h5_store_bk data/processed/h5_store_sc_bk
              data/hasmig_mhcs/burial_score_output.tar.gz
              src/model/h5_test.py src/model_multimer_1/run_normprobe.sh)
  for p in "${must_refuse[@]}"; do
    if guard "$p" >/dev/null 2>&1; then
      printf "%sSELF-TEST FAIL: guard ALLOWED protected '%s'%s\n" "$RED" "$p" "$OFF"; fails=1
    fi
  done
  for p in "${must_allow[@]}"; do
    [[ -e "$p" ]] || continue
    if ! guard "$p" >/dev/null 2>&1; then
      printf "%sSELF-TEST FAIL: guard REFUSED removable '%s'%s\n" "$RED" "$p" "$OFF"; fails=1
    fi
  done
  if (( fails )); then printf "%sself-test FAILED%s\n" "$RED" "$OFF"; exit 1; fi
  printf "%sself-test PASSED%s: guard refuses all %d protected paths and permits all %d removable ones\n" \
         "$GRN" "$OFF" "${#must_refuse[@]}" "${#must_allow[@]}"
  exit 0
fi

add() {           # add <path>  -> queue it for the current group
  local p="$1"
  [[ -e "$p" ]] || return 0
  guard "$p" || { GROUP_BLOCKED=1; return 1; }
  local kb; kb=$(du -sk "$p" 2>/dev/null | cut -f1); kb=${kb:-0}
  PENDING+=("$kb|$p")
}

human() { local kb=$1
  if   (( kb >= 1048576 )); then printf "%.1f G" "$(echo "$kb/1048576"|bc -l)"
  elif (( kb >= 1024 ));    then printf "%.0f M" "$(echo "$kb/1024"|bc -l)"
  else                           printf "%d K" "$kb"; fi; }

run_group() {     # run_group <name> <description>
  local name="$1" desc="$2"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then PENDING=(); return 0; fi
  if (( ${#PENDING[@]} == 0 )); then
    printf "%s[%s]%s nothing to do\n" "$DIM" "$name" "$OFF"; return 0
  fi
  local sum=0 e
  for e in "${PENDING[@]}"; do sum=$(( sum + ${e%%|*} )); done
  printf "\n%s=== %s ===%s  %s\n" "$YEL" "$name" "$OFF" "$desc"
  for e in "${PENDING[@]}"; do printf "   %8s  %s\n" "$(human "${e%%|*}")" "${e#*|}"; done
  printf "   %s--------%s  %s total\n" "$DIM" "$OFF" "$(human $sum)"
  if (( GROUP_BLOCKED )); then
    printf "%s   group had a blocked path; skipping the whole group%s\n" "$RED" "$OFF"
    PENDING=(); GROUP_BLOCKED=0; return 0
  fi
  if (( ! APPLY )); then PENDING=(); return 0; fi
  if (( ! ASSUME_YES )); then
    read -r -p "   delete these? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { printf "   skipped\n"; PENDING=(); return 0; }
  fi
  for e in "${PENDING[@]}"; do
    local p="${e#*|}"
    guard "$p" || continue                       # re-check right before rm
    if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
      git rm -r -q --cached "$p" && rm -rf "$p"  # tracked: history keeps a copy
    else
      rm -rf "$p"
    fi
  done
  TOTAL_KB=$(( TOTAL_KB + sum ))
  printf "%s   removed %s%s\n" "$GRN" "$(human $sum)" "$OFF"
  PENDING=(); GROUP_BLOCKED=0
}

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
echo "PMGen2 cleanup  —  $ROOT"
(( APPLY )) && echo "${RED}MODE: APPLY (will delete)${OFF}" \
            || echo "${GRN}MODE: dry run (nothing will be deleted)${OFF}"

if command -v squeue >/dev/null 2>&1; then
  nj=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
  if (( nj > 0 )); then
    echo "${RED}PREFLIGHT FAIL: $nj SLURM job(s) still running — a training job may be${OFF}"
    echo "${RED}writing into checkpoints_mm1/ or the H5 stores. Wait for them to finish.${OFF}"
    squeue -u "$USER"
    (( APPLY )) && exit 1
  fi
fi

for f in src/model/utils.py src/pdb/parse.py src/model_multimer_1/model.py \
         outputs/data_exploration/per_structure.csv \
         checkpoints_mm1/ARCHIVE/stage2b_best.pt \
         data/processed/h5_store_sc/index.csv \
         data/processed/h5_store_hasmig/index.csv; do
  [[ -e "$f" ]] || { echo "${RED}PREFLIGHT FAIL: required file missing: $f${OFF}"; exit 1; }
done
echo "${GRN}preflight ok${OFF}: required inputs present, no jobs running"

GROUP_BLOCKED=0

# ---------------------------------------------------------------------------
# 1. H5 backup stores — only after re-proving the superset relation NOW
# ---------------------------------------------------------------------------
echo
echo "verifying the live H5 stores are strict supersets of the backups..."
VERDICT=$("$PY" - <<'PYV'
import glob, sys
import pandas as pd
def ids(d):
    s = set()
    for f in glob.glob(f"data/processed/{d}/*index.csv"):
        s |= set(pd.read_csv(f).id.astype(str))
    return s
def fields(d):
    import h5py
    f = sorted(glob.glob(f"data/processed/{d}/*.h5"))[0]
    with h5py.File(f, "r") as h:
        k = next(iter(h)); return set(h[k].keys())
ok = []
for live, bk in [("h5_store_sc", "h5_store_sc_bk"),
                 ("h5_store_sc", "h5_store_bk"),
                 ("h5_store_hasmig", "h5_store_hasmig_bk")]:
    try:
        li, bi = ids(live), ids(bk)
        lf, bf = fields(live), fields(bk)
    except Exception as e:
        print(f"SKIP {bk}: {e}", file=sys.stderr); continue
    missing = bi - li
    lost = bf - lf
    if not missing and not lost:
        ok.append(bk)
        print(f"  OK  {bk:22s} {len(bi):>7,} ids, all present in {live}; "
              f"fields {sorted(lf-bf)} added, none lost", file=sys.stderr)
    else:
        print(f"  NO  {bk}: {len(missing)} ids and {len(lost)} fields NOT in {live} "
              f"-- keeping it", file=sys.stderr)
print(" ".join(ok))
PYV
)
for bk in $VERDICT; do add "data/processed/$bk"; done
run_group h5-backups "superseded H5 generations (verified strict subsets just now)"

# ---------------------------------------------------------------------------
# 2. the redundant hasmig tarball
# ---------------------------------------------------------------------------
TAR="data/hasmig_mhcs/burial_score_output.tar.gz"
DIR="data/hasmig_mhcs/burial_score_output"
if [[ -f "$TAR" && -d "$DIR" ]]; then
  n_tar=$(tar tf "$TAR" 2>/dev/null | grep -c '\.zip$')
  n_dir=$(find "$DIR" -maxdepth 1 -name '*.zip' | wc -l)
  echo
  echo "hasmig archive: tarball holds $n_tar zips, extracted dir holds $n_dir"
  if (( n_dir >= n_tar && n_tar > 0 )); then
    add "$TAR"
  else
    echo "${RED}   extracted dir is INCOMPLETE — keeping the tarball${OFF}"
  fi
fi
run_group hasmig-tar "tarball whose contents are already extracted beside it"

# ---------------------------------------------------------------------------
# 3. AFDB corpus (only src/model4 reads it, and model4 is retired)
# ---------------------------------------------------------------------------
if ! grep -rl "data/afdb\|plddt_dataset" src --include='*.py' --include='*.sbatch' \
        --include='*.sh' 2>/dev/null | grep -qv '^src/model4/'; then
  add data/afdb
else
  echo; echo "${RED}data/afdb is referenced outside src/model4 — keeping it${OFF}"
  grep -rl "data/afdb\|plddt_dataset" src --include='*.py' --include='*.sbatch' | grep -v '^src/model4/'
fi
run_group afdb "AFDB pretraining corpus — only src/model4 (retired) reads it"

# ---------------------------------------------------------------------------
# 4. retired model generations' checkpoints
# ---------------------------------------------------------------------------
add checkpoints
add checkpoints_model2; add checkpoints_model3; add checkpoints_model4
add checkpoints_mm1_bk
add tmp
run_group old-checkpoints "pre-mm1 model checkpoints + the leaky pre-fix mm1 backups"

# ---------------------------------------------------------------------------
# 5. dead mm1 probe runs.  ARCHIVE, mm1_stage2 and mm1_stage2b are NOT here.
# ---------------------------------------------------------------------------
for d in mm1_s1_full mm1_s1_short mm1_s1_long mm1_s1_fix mm1_s1_final \
         trig_A trig_B trig_C mm1_normprobe mm1_archprobe mm1_shortprobe \
         mm1_lrprobe_lr5e4 mm1_lrprobe_lr1e4 mm1_lrprobe_fix5e4 \
         mm1_guardsmoke mm1_bs12 mm1_long24 mm1_stage1_sc snapshots; do
  add "checkpoints_mm1/$d"
done
for f in checkpoints_mm1/*.pt; do add "$f"; done
run_group mm1-probes "mm1 runs that died or were probes (ARCHIVE/ + stage2/ + stage2b/ kept)"

# ---------------------------------------------------------------------------
# 6. retired code.  Tracked files stay recoverable from git history.
# ---------------------------------------------------------------------------
add src/model_2; add src/model_3; add src/model4
for f in data_test.py encoder_test.py h5_test.py overfit_test.py train_test.py \
         train_grid.sbatch train_grid_1fold.sbatch; do add "src/model/$f"; done
for f in run_normprobe.sh run_archprobe.sh run_shortprobe.sh run_trigger_probe.sh \
         run_lr_probe.sh run_stage1_full.sh run_stage1_long.sh run_stage1_short.sh \
         run_stage1_sidechain.sh run_stage1_final.sh run_stage1_long24.sh \
         run_A_resume.sh run_B_resume.sh; do add "src/model_multimer_1/$f"; done
add tmp.txt
run_group dead-code "retired model generations, dev smoke tests, dead probe launchers"

# ---------------------------------------------------------------------------
# 7. logs from retired generations.  logs/mm1 and logs/flow are kept.
# ---------------------------------------------------------------------------
for d in distill diff af3 m4 m4_2 eval; do add "logs/$d"; done
add logs/dataexp_28588668.out; add logs/dataexp_28588668.err
run_group old-logs "SLURM logs from retired generations (logs/mm1 and logs/flow kept)"

# ---------------------------------------------------------------------------
# 8. __pycache__
# ---------------------------------------------------------------------------
# __pycache__ is generated, never source, so it is exempt from the PROTECTED
# containment rule -- but ONLY when the basename really is __pycache__ and the
# path really is inside this repo.
PYC_KB=0; declare -a PYC_DIRS=()
while IFS= read -r d; do
  [[ "$(basename "$d")" == "__pycache__" ]] || continue
  abs="$(readlink -f "$d")"; [[ "$abs" == "$ROOT"/* ]] || continue
  kb=$(du -sk "$d" 2>/dev/null | cut -f1); PYC_KB=$(( PYC_KB + ${kb:-0} ))
  PYC_DIRS+=("$d")
done < <(find src openfold -name __pycache__ -type d 2>/dev/null)
if [[ -z "$ONLY" || "$ONLY" == "pycache" ]] && (( ${#PYC_DIRS[@]} )); then
  printf "\n%s=== pycache ===%s  compiled bytecode (%d dirs, %s)\n" \
         "$YEL" "$OFF" "${#PYC_DIRS[@]}" "$(human $PYC_KB)"
  if (( APPLY )); then
    ans=y
    (( ASSUME_YES )) || read -r -p "   delete these? [y/N] " ans
    if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
      for d in "${PYC_DIRS[@]}"; do rm -rf "$d"; done
      TOTAL_KB=$(( TOTAL_KB + PYC_KB ))
      printf "%s   removed %s%s\n" "$GRN" "$(human $PYC_KB)" "$OFF"
    else printf "   skipped\n"; fi
  fi
fi

# ---------------------------------------------------------------------------
# POST-CLEANUP VERIFICATION
# ---------------------------------------------------------------------------
echo
if (( APPLY )); then
  echo "${YEL}=== verifying the pipeline still imports and the model still loads ===${OFF}"
  "$PY" - <<'PYC'
import sys
from pathlib import Path
R = Path(".").resolve()
sys.path[:0] = [str(R/"openfold"), str(R/"src"/"model"), str(R/"src"/"model_multimer_1")]
import torch, utils as m1, model as MM                      # noqa
net = MM.MultimerModel(n_trunk=3, device="cpu", pep_frames="identity",
                       angle_input="layernorm")
net.set_stage(2)
ck = torch.load("checkpoints_mm1/ARCHIVE/stage2b_best.pt",
                map_location="cpu", weights_only=False)
missing, unexpected = net.load_state_dict(ck["trainable"], strict=False)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
print(f"  OK  mm1 model builds and stage2b_best.pt loads (gstep {ck['global_step']})")
import pandas as pd
for f in ["outputs/data_exploration/per_structure.csv",
          "data/processed/h5_store_sc/index.csv",
          "data/processed/h5_store_hasmig/index.csv",
          "benchmark/pRMSD_benchmark.tsv"]:
    assert Path(f).exists(), f
print("  OK  training inputs and the benchmark TSV are present")
sys.path.insert(0, str(R/"src"/"pdb"))
import parse                                                # noqa
print("  OK  src/pdb/parse.py imports (shared with the flow model)")
PYC
  rc=$?
  if (( rc == 0 )); then echo "${GRN}post-cleanup verification PASSED${OFF}"
  else echo "${RED}post-cleanup verification FAILED (rc=$rc) — investigate before committing${OFF}"; fi
  echo
  echo "flow-model territory, untouched:"
  for p in src/flow_model checkpoints_flow logs/flow data/processed/flow_canonical; do
    printf "   %-34s %s\n" "$p" "$([[ -e $p ]] && echo present || echo "${RED}MISSING${OFF}")"
  done
  echo
  echo "${GRN}total reclaimed: $(human $TOTAL_KB)${OFF}"
else
  echo "${GRN}dry run complete — nothing was deleted. Re-run with --apply to act.${OFF}"
fi
