import sys
from pathlib import Path

# --- openfold on path (package lives at REPO_ROOT/openfold/openfold) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_DIR = REPO_ROOT / "params" / "alphafold"
sys.path.insert(0, str(REPO_ROOT / "openfold"))

import torch
import torch.nn as nn
from openfold.config import model_config
from openfold.model.structure_module import StructureModule
from openfold.model.heads import PerResidueLDDTCaPredictor, TMScoreHead


class FrozenFold(nn.Module):
    """Frozen AF2 structure module + pLDDT/PAE heads. Encoder feeds (s, z)."""

    def __init__(self, cfg):
        super().__init__()
        self.sm    = StructureModule(**cfg["model"]["structure_module"])
        self.plddt = PerResidueLDDTCaPredictor(**cfg["model"]["heads"]["lddt"])
        self.tm    = TMScoreHead(**cfg["model"]["heads"]["tm"])
        for m in (self.sm, self.plddt, self.tm):
            m.eval()
            m.requires_grad_(False)

    def forward(self, s, z, aatype, seq_mask):
        out = self.sm({"single": s, "pair": z}, aatype, mask=seq_mask)
        ca  = out["positions"][-1][..., 1, :]   # last block, atom14 Cα = index 1
        return ca, self.plddt(out["single"]), self.tm(z)


def load_frozen_fold(model_name="model_2_ptm", params_dir=PARAMS_DIR, device="cpu"):
    """Reload the frozen stack from extracted weights. Use in train/inference."""
    cfg = model_config(model_name)
    ff = FrozenFold(cfg)
    ff.sm.load_state_dict(torch.load(params_dir / "sm_af2.pt", weights_only=True))
    ff.plddt.load_state_dict(torch.load(params_dir / "plddt_af2.pt", weights_only=True))
    ff.tm.load_state_dict(torch.load(params_dir / "pae_af2.pt", weights_only=True))
    return ff.to(device).eval()