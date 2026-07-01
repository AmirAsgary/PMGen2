#!/bin/bash -l
# Download DeepMind's public AlphaFold params and keep only the multimer model we need
# (params_model_1_multimer_v3.npz), into params/alphafold/. The tar is ~5.3 GB; we
# extract a single ~1 GB npz and delete the tar.
#
#   bash src/model_multimer_1/download_multimer_weights.sh
# Then extract the IPA + pLDDT-head weights:
#   $PY src/model_multimer_1/extract_multimer_weights.py
set -euo pipefail

DEST=${DEST:-params/alphafold}
MODEL=${MODEL:-model_1_multimer_v3}
URL=${URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}

mkdir -p "$DEST"
if [[ -f "$DEST/params_${MODEL}.npz" ]]; then
    echo "already have $DEST/params_${MODEL}.npz"; exit 0
fi
echo "downloading AlphaFold params tar (~5.3 GB) ..."
curl -L --fail -o "$DEST/af_params.tar" "$URL"
echo "extracting params_${MODEL}.npz ..."
tar -xvf "$DEST/af_params.tar" -C "$DEST" "params_${MODEL}.npz"
rm -f "$DEST/af_params.tar"
echo "done -> $DEST/params_${MODEL}.npz"
