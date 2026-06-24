"""
model_3 feature builder: our H5 batch -> single-sequence AlphaFold input features.

model_3 runs the REAL pretrained AF2 Evoformer on a depth-1 "MSA" (just the paired
peptide+MHC query sequence), no templates, no extra MSA. This module turns the
model_1/2 per-residue fields (``aatype``, ``residue_index``, ``anchor``,
``segment_id``, ``seq_mask``) into exactly the tensors AF2's ``InputEmbedder`` +
``EvoformerStack`` consume, in the layout the pretrained weights expect.

Layout (verified against openfold ``data/data_transforms.py:make_msa_feat`` and
``residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE`` — after AF2's restype remap the
query MSA row uses the SAME restype order as our ``aatype``, so a plain one-hot is
correct):
  target_feat[B,N,22] = [ has_break(1) , one_hot(aatype,21) ]
  msa_feat[B,1,N,49]  = [ one_hot(aatype,23) , has_del(1)=0 , del_val(1)=0 ,
                          cluster_profile(23)=one_hot(aatype,23) , clus_del_mean(1)=0 ]

The ~200 MHC->peptide gap that PMGen baked into ``residue_index`` is the AF2 monomer
"gap trick" for complexes (relpos clips at ±32 -> a >=200 jump reads as a chain
break). We pass ``residue_index`` straight through; ``anchor_fn`` (the model_1
``anchor_canonical_resindex``) renumbers the peptide from its anchors, *preserving*
that gap and adding the register on top.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def _has_break(segment_id: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """[B,N] 1.0 at the residue BEFORE a segment change (the MHC->peptide boundary),
    else 0.0 — AF2's ``between_segment_residues`` signal."""
    m = seq_mask.bool()
    out = torch.zeros_like(seq_mask, dtype=torch.float32)
    diff = (segment_id[:, :-1] != segment_id[:, 1:]) & m[:, :-1] & m[:, 1:]
    out[:, :-1] = diff.float()
    return out


def build_af2_features(aatype: torch.Tensor, residue_index: torch.Tensor,
                       anchor: torch.Tensor, segment_id: torch.Tensor,
                       seq_mask: torch.Tensor,
                       anchor_fn: Optional[Callable] = None
                       ) -> Dict[str, torch.Tensor]:
    """Build the depth-1 AF2 input dict. ``aatype`` is in restype_order 0..20 (matches
    AF2). If ``anchor_fn`` is given (model_1's ``anchor_canonical_resindex``) it is
    applied to ``residue_index`` first so the peptide register reaches AF2 via relpos."""
    B, N = aatype.shape
    dev = aatype.device
    ri = residue_index
    if anchor_fn is not None:
        ri = anchor_fn(residue_index, anchor, seq_mask, segment_id)

    aa = aatype.clamp(0, 20).long()                      # padding (0) is masked later
    aa21 = F.one_hot(aa, 21).to(seq_mask.dtype)          # [B,N,21]
    has_break = _has_break(segment_id, seq_mask)         # [B,N]
    target_feat = torch.cat([has_break[..., None], aa21], dim=-1)        # [B,N,22]

    aa23 = F.one_hot(aa, 23).to(seq_mask.dtype)          # [B,N,23] (classes 21,22 = 0)
    msa_1hot = aa23[:, None]                              # [B,1,N,23]
    zero = torch.zeros(B, 1, N, 1, device=dev, dtype=seq_mask.dtype)
    # [msa_1hot, has_deletion, deletion_value, cluster_profile(=msa_1hot), clus_del_mean]
    msa_feat = torch.cat([msa_1hot, zero, zero, msa_1hot, zero], dim=-1)  # [B,1,N,49]

    return {
        "target_feat": target_feat,
        "msa_feat": msa_feat,
        "residue_index": ri.long(),
        "seq_mask": seq_mask.to(seq_mask.dtype),
        "msa_mask": seq_mask[:, None].to(seq_mask.dtype),                 # [B,1,N]
        "pair_mask": (seq_mask[:, :, None] * seq_mask[:, None, :]),       # [B,N,N]
    }
