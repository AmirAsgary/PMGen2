"""
model_3 (AF-Evo distill): the REAL pretrained AF2 Evoformer, truncated to its first
K blocks and mostly frozen, feeding the frozen AF2 structure module + pLDDT/PAE heads.
Only a thin slice (the last kept Evoformer block, ``trainable`` %) is fine-tuned, so
we *distill from a portion of the Evoformer* rather than learning an encoder from
scratch (model_1) — the representation the SM needs provably exists, we just adapt it.

Inputs are the SAME per-residue fields as model_1/2 (single-sequence, anchors via
residue_index renumbering, no MSA, no templates). ``EvoDistillModel`` mirrors
``DistillModel``'s forward signature so it reuses model_1's loss + training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
sys.path.insert(0, str(_ROOT / "openfold"))            # openfold package
sys.path.insert(0, str(_SRC / "model"))                # model_1 utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir (featurize)

import utils as m1                                      # noqa: E402  (model_1 utils)
import featurize as feat                                # noqa: E402
from openfold.config import model_config                # noqa: E402
from openfold.model.model import AlphaFold              # noqa: E402
from openfold.utils.import_weights import import_jax_weights_  # noqa: E402

NPZ = _ROOT / "params" / "alphafold" / "params_model_2_ptm.npz"


class EvoDistillModel(nn.Module):
    """Pretrained AF2 input-embedder + first-K Evoformer blocks (last block's last
    ``trainable`` % unfrozen) -> frozen AF2 structure module + pLDDT/PAE heads.

    All the heavy weights come from ONE ``import_jax_weights_`` of the monomer npz;
    we keep ``input_embedder`` + truncated ``evoformer`` + ``structure_module`` +
    ``aux_heads.{plddt,tm}`` and drop templates / extra-MSA / recycling (unused)."""

    def __init__(self, evo_layers: int = 3, trainable: float = 10.0,
                 model_name: str = "model_2_ptm", npz: Path = NPZ,
                 device: str = "cpu"):
        super().__init__()
        cfg = model_config(model_name)
        # Build the FULL model and import the full npz cleanly (avoids any
        # disable-then-import key mismatch), then keep only the pieces we need.
        af = AlphaFold(cfg)
        import_jax_weights_(af, str(npz), version=model_name)

        self.input_embedder = af.input_embedder
        self.evoformer = af.evoformer
        # truncate to the first K blocks (keeps blocks 0..K-1 = correct AF2 weights)
        K = int(evo_layers)
        self.evoformer.blocks = nn.ModuleList(list(self.evoformer.blocks)[:K])
        self.evoformer.blocks_per_ckpt = 1             # gradient checkpoint each block
        self.structure_module = af.structure_module
        self.plddt = af.aux_heads.plddt
        self.pae = af.aux_heads.tm                     # ptm TM/PAE head
        del af                                         # free templates/extra-msa/etc.

        # freeze everything, then unfreeze the last `trainable` % of the LAST kept
        # Evoformer block (reuse model_1's last-pct unfreezer).
        self.requires_grad_(False)
        self.unfrozen = m1.DistillModel._unfreeze_last_pct(
            self.evoformer.blocks[K - 1], trainable)
        self.evo_layers = K
        self.recycles = 0                              # train loop reads this (no recycle)
        self.to(device)
        self.train(False)                              # start in eval; train() keeps it so

    def train(self, mode: bool = True):
        """Keep the pretrained AF2 submodules in EVAL (no dropout / stochasticity) even
        during training — only the unfrozen *parameters* learn (via requires_grad,
        which is independent of train/eval). Mirrors model_1's frozen-stack handling
        and keeps the forward deterministic."""
        super().train(mode)
        for mod in (self.input_embedder, self.evoformer, self.structure_module,
                    self.plddt, self.pae):
            mod.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor], return_frames: bool = False,
                num_recycles: Optional[int] = None):
        """-> (ca[B,N,3], plddt_logits[B,N,50], pae_logits[B,N,N,64]) and, if
        ``return_frames``, the SM backbone-frame trajectory for FAPE. ``num_recycles``
        is ignored (single pass); kept for signature parity with DistillModel."""
        feats = feat.build_af2_features(
            batch["aatype"], batch["residue_index"], batch["anchor"],
            batch["segment_id"], batch["seq_mask"],
            anchor_fn=m1.anchor_canonical_resindex)
        m, z = self.input_embedder(feats["target_feat"], feats["residue_index"],
                                   feats["msa_feat"])
        m, z, s = self.evoformer(m, z, feats["msa_mask"], feats["pair_mask"])
        # frozen SM/heads stay in the autograd graph (grad flows THROUGH them into the
        # trainable Evoformer slice) — same as model_1.
        out = self.structure_module({"single": s, "pair": z}, batch["aatype"],
                                    mask=batch["seq_mask"])
        ca = out["positions"][-1][..., 1, :]
        plddt_logits = self.plddt(out["single"])
        pae_logits = self.pae(z)
        if return_frames:
            return ca, plddt_logits, pae_logits, out["frames"]
        return ca, plddt_logits, pae_logits

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        """state_dict of only the unfrozen params — the whole checkpoint (the rest is
        reloaded from the npz at construction)."""
        return {n: p.detach().cpu() for n, p in self.named_parameters()
                if p.requires_grad}
