"""
Extract the AlphaFold-Multimer pieces model_multimer_1 needs, into params/alphafold/
as small .pt state_dicts (mirrors src/afbuild/build.py for the monomer):

  input_embedder_mm.pt  -- InputEmbedderMultimer (head-1: seq -> single/pair)
  sm_mm.pt              -- multimer StructureModule (frozen structure predictor +
                           the IPA weights we reuse for head-2 / the trunk IPAs)
  plddt_mm.pt           -- multimer per-residue pLDDT head (frozen)

Run once on the cluster (needs openfold + torch), after download_multimer_weights.sh:
  $PY src/model_multimer_1/extract_multimer_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openfold"))
from openfold.config import model_config                 # noqa: E402
from openfold.model.model import AlphaFold                # noqa: E402
from openfold.utils.import_weights import import_jax_weights_  # noqa: E402

MODEL = "model_1_multimer_v3"
PARAMS = ROOT / "params" / "alphafold"
NPZ = PARAMS / f"params_{MODEL}.npz"


def main():
    assert NPZ.exists(), f"missing {NPZ} — run download_multimer_weights.sh first"
    cfg = model_config(MODEL)
    print(f"building AlphaFold({MODEL}) and importing weights ...")
    model = AlphaFold(cfg)
    import_jax_weights_(model, str(NPZ), version=MODEL)

    torch.save(model.input_embedder.state_dict(), PARAMS / "input_embedder_mm.pt")
    torch.save(model.structure_module.state_dict(), PARAMS / "sm_mm.pt")
    torch.save(model.aux_heads.plddt.state_dict(), PARAMS / "plddt_mm.pt")
    print("saved input_embedder_mm.pt, sm_mm.pt, plddt_mm.pt ->", PARAMS)
    # also dump the sub-configs model_multimer_1 will need to reconstruct the modules
    import json
    keep = {
        "input_embedder": dict(cfg["model"]["input_embedder"]),
        "structure_module": dict(cfg["model"]["structure_module"]),
        "lddt": dict(cfg["model"]["heads"]["lddt"]),
    }
    (PARAMS / "mm_subconfig.json").write_text(json.dumps(keep, indent=2, default=str))
    print("saved mm_subconfig.json")


if __name__ == "__main__":
    main()
