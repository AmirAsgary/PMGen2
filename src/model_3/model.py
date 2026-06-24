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
                 device: str = "cpu", unfreeze_sm: float = 0.0,
                 unfreeze_plddt: float = 0.0, unfreeze_pae: float = 0.0,
                 full_evoformer: bool = False):
        super().__init__()
        cfg = model_config(model_name)
        # Build the FULL model and import the full npz cleanly (avoids any
        # disable-then-import key mismatch), then keep only the pieces we need.
        af = AlphaFold(cfg)
        import_jax_weights_(af, str(npz), version=model_name)

        self.input_embedder = af.input_embedder
        self.evoformer = af.evoformer
        # keep the first K blocks (K=48 -> the FULL Evoformer)
        K = int(evo_layers)
        self.evoformer.blocks = nn.ModuleList(list(self.evoformer.blocks)[:K])
        self.structure_module = af.structure_module
        self.plddt = af.aux_heads.plddt
        self.pae = af.aux_heads.tm                     # ptm TM/PAE head
        del af                                         # free templates/extra-msa/etc.

        self.full_evoformer = bool(full_evoformer)
        _uf = m1.DistillModel._unfreeze_last_pct
        self.requires_grad_(False)
        if self.full_evoformer:
            # FULL retrain: input embedder + every kept Evoformer block are trainable.
            # Gradient checkpointing ON (essential at K=48) — and it WORKS here because
            # the now-trainable embedder output requires grad (the reentrant checkpoint
            # only breaks when its inputs don't, which was the frozen-prefix case).
            self.input_embedder.requires_grad_(True)
            self.evoformer.requires_grad_(True)
            self.evoformer.blocks_per_ckpt = 1
            self.unfrozen = {
                "input_embedder": 1.0, "evoformer_all": 1.0,
                "sm": _uf(self.structure_module, unfreeze_sm),
                "plddt": _uf(self.plddt, unfreeze_plddt),
                "pae": _uf(self.pae, unfreeze_pae)}
        else:
            # lightweight: frozen prefix (run under no_grad in forward), only the last
            # block's last `trainable` % + optional SM/head tails. No checkpointing
            # (reentrant checkpoint would detach the trainable block from the no_grad
            # prefix); cheap enough without it.
            self.evoformer.blocks_per_ckpt = None
            self.unfrozen = {
                "evo_last_block": _uf(self.evoformer.blocks[K - 1], trainable),
                "sm": _uf(self.structure_module, unfreeze_sm),
                "plddt": _uf(self.plddt, unfreeze_plddt),
                "pae": _uf(self.pae, unfreeze_pae)}
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
        if self.full_evoformer:
            # everything trainable -> standard checkpointed Evoformer forward.
            m, z = self.input_embedder(feats["target_feat"], feats["residue_index"],
                                       feats["msa_feat"])
            m, z, s = self.evoformer(m, z, feats["msa_mask"], feats["pair_mask"])
        else:
            # Run the FROZEN prefix (input embedder + all but the last Evoformer block)
            # under no_grad: nothing before the trainable last block needs a backward,
            # so this skips that compute AND frees the prefix activations -> far less
            # memory (=> a much larger batch) and a faster step.
            with torch.no_grad():
                m, z = self.input_embedder(feats["target_feat"], feats["residue_index"],
                                           feats["msa_feat"])
                blocks = self.evoformer._prep_blocks(
                    m=m, z=z, chunk_size=None, use_deepspeed_evo_attention=False,
                    use_cuequivariance_attention=False,
                    use_cuequivariance_multiplicative_update=False, use_lma=False,
                    use_flash=False, msa_mask=feats["msa_mask"],
                    pair_mask=feats["pair_mask"], inplace_safe=False, _mask_trans=True)
                for blk in blocks[:-1]:
                    m, z = blk(m, z)
            # the last block holds the only trainable params -> run WITH grad; the
            # frozen SM/heads stay in the graph so grad flows back into that block.
            m, z = blocks[-1](m, z)
            s = self.evoformer.linear(m[..., 0, :, :])
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
