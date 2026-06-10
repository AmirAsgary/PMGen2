#!/usr/bin/env bash
# ============================================================================
# Install the `pmgen2` conda env for PMGen-v2 distillation (training + HDF5
# preprocessing). Lean by design: torch (CUDA) + numpy/scipy/pandas/h5py/
# biopython/ml-collections/einops. OpenFold is used from ./openfold on
# PYTHONPATH (no pip install, no CUDA-kernel build).
#
# Usage:
#   bash installation.sh [ENV_NAME] [CUDA_TAG]
#     ENV_NAME  default: pmgen2
#     CUDA_TAG  default: cu121   (match the cluster's NVIDIA driver: cu118/cu121/cu124)
#   Override torch version with TORCH_VERSION=... (default 2.5.1).
#
# On HPC you may need `module load anaconda` (or miniforge) first.
# ============================================================================
set -euo pipefail

ENV_NAME="${1:-pmgen2}"
CUDA_TAG="${2:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v mamba >/dev/null 2>&1; then CONDA=mamba; else CONDA=conda; fi
echo "[install] $CONDA | env=$ENV_NAME | torch==${TORCH_VERSION}+${CUDA_TAG}"

# 1) create (or update) the conda env from the spec
if $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[install] env '$ENV_NAME' exists -> updating"
  $CONDA env update -n "$ENV_NAME" -f "$HERE/environment.yml"
else
  $CONDA env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

# resolve the env's python by absolute path (robust on non-interactive shells)
CONDA_BASE="$(conda info --base | awk 'END {print $NF}')"
ENV_PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
[ -x "$ENV_PY" ] || { echo "[install] ERROR: $ENV_PY not found"; exit 1; }
echo "[install] env python: $ENV_PY"

# 2) CUDA PyTorch from the official index (bundles its own CUDA runtime)
"$ENV_PY" -m pip install --upgrade pip

if [ -z "$TORCH_VERSION" ]; then
    echo "[install] Installing latest PyTorch for $CUDA_TAG"
    "$ENV_PY" -m pip install "torch" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
    echo "[install] Installing specifically PyTorch == $TORCH_VERSION for $CUDA_TAG"
    "$ENV_PY" -m pip install "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# Install extra packages into the env (route via the env's python, not the
# active shell's `pip`). tqdm is also in environment.yml; dm-tree/modelcif are
# optional OpenFold extras not required by the distillation path.
"$ENV_PY" -m pip install dm-tree modelcif tqdm

# 3) verify
"$ENV_PY" - <<'PYEOF'
import torch, numpy, scipy, pandas, h5py, Bio, ml_collections, einops
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "| cuda", torch.version.cuda)
print("deps OK | numpy", numpy.__version__, "| h5py", h5py.__version__,
      "| biopython", Bio.__version__)
PYEOF

echo
echo "[install] done. OpenFold is used from ./openfold on PYTHONPATH (no install)."
echo "[install] smoke test (uses the 15 dummy examples, needs a GPU):"
echo "          $ENV_PY src/model/encoder_test.py"
echo "          $ENV_PY src/model/train.py --dummy --variant 7 --epochs 5 --bs 3"