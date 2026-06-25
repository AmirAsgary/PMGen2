"""
model4 / plddt_model — architecture.

A single-row AlphaFold-Evoformer-style network that maps a protein *sequence*
(plus optional anchor / spatial-proximity hints) to per-residue pLDDT, while
learning reusable ``single`` (per-residue) and ``pair`` (per-residue-pair)
representations.  Stage-1 pretraining is on the AFDB monomer set; see
``src/model4/train.py`` and the plan under ``.claude/plans``.

Reused, never reimplemented (OpenFold, vendored at ``PMGen2/openfold``):
  Linear, LayerNorm, Attention, TriangleMultiplication{Outgoing,Incoming},
  TriangleAttention{StartingNode,EndingNode}, OuterProductMean, PairTransition,
  PerResidueLDDTCaPredictor (pLDDT head; weights ``params/alphafold/plddt_af2.pt``).

The ``SingleUpdate`` block (single self-attention biased by pair + transition) is
copied verbatim from ``src/model/utils.py`` to keep model4 decoupled from the
pMHC-specific frozen-fold machinery in that module.

Env: pmgen2 (~/miniforge3/envs/pmgen2/bin/python).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Repo wiring: expose the vendored OpenFold package.
# --------------------------------------------------------------------------- #
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "openfold").is_dir() and (parent / "src" / "afbuild").is_dir():
            return parent
    raise RuntimeError("could not locate PMGen2 repo root (openfold + src/afbuild)")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT / "openfold") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "openfold"))

from openfold.model.primitives import Linear, LayerNorm, Attention          # noqa: E402
from openfold.model.triangular_multiplicative_update import (               # noqa: E402
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.model.triangular_attention import (                           # noqa: E402
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.model.outer_product_mean import OuterProductMean              # noqa: E402
from openfold.model.pair_transition import PairTransition                   # noqa: E402
from openfold.model.heads import PerResidueLDDTCaPredictor                  # noqa: E402
from openfold.utils.loss import compute_plddt                              # noqa: E402


# --------------------------------------------------------------------------- #
# Vocab / feature constants (kept in sync with src/model4/data.py).
# --------------------------------------------------------------------------- #
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXBZJU"   # 25 letters, stored token 0=A .. 24=U
PAD_TOKEN = 0
N_AA = len(AA_ALPHABET)                     # 25  -> real tokens 1..25 (shifted +1)
SEP_TOKEN = N_AA + 1                         # 26
MASK_TOKEN = N_AA + 2                        # 27
VOCAB_SIZE = N_AA + 3                         # 28  (pad, 25 aa, sep, mask)

MAX_SEGMENTS = 16                            # simulated chains (domains) per crop
SEG_SEP = MAX_SEGMENTS                       # reserved segment id for SEP / pad rows
N_PE_BINS = 16                              # 0..13 contact dist, 14 non-contact, 15 null
PE_NULL_BIN = 15
N_ANCHOR_STATES = 3                          # 0=Unknown, 1=NoAnchor, 2=AnchorKnown
N_DIST_BINS = 16                            # distogram: 15 buckets over [0,8A) + 1 ">=8A"
N_DIST_FAR = N_DIST_BINS - 1                 # bin index for non-contact (>=8A) pairs
PLDDT_HEAD_DIM = 384                         # AF pLDDT head input dim (fixed by weights);
                                            # the body runs narrower and up-projects to this


def _chain_relpos_onehot(residue_index: torch.Tensor, segment_id: torch.Tensor,
                         max_offset: int) -> torch.Tensor:
    """AF-Multimer-style relative position: clipped |i-j| WITHIN a segment, plus a
    dedicated 'different-segment' bin for cross-chain pairs. residue_index is reset
    per segment, so same-segment offsets are true sequence separations and cross-
    segment pairs are explicitly distinguished (key for the cross-chain anchors).
    Returns [B, N, N, 2*max_offset+2]."""
    diff = residue_index[:, :, None] - residue_index[:, None, :]
    binned = diff.clamp(-max_offset, max_offset) + max_offset           # 0..2k
    same = segment_id[:, :, None] == segment_id[:, None, :]             # [B,N,N]
    diff_idx = 2 * max_offset + 1                                       # extra bin
    binned = torch.where(same, binned, torch.full_like(binned, diff_idx))
    return F.one_hot(binned, num_classes=2 * max_offset + 2).float()


# --------------------------------------------------------------------------- #
# Featurizer: tokens/segments -> s0 ; outer-sum + relpos + anchor feats -> z0.
# --------------------------------------------------------------------------- #
class Featurizer(nn.Module):
    def __init__(self, c_s: int, c_z: int, max_offset: int = 32) -> None:
        super().__init__()
        self.max_offset = max_offset
        self.tok_emb = nn.Embedding(VOCAB_SIZE, c_s, padding_idx=PAD_TOKEN)
        self.seg_emb = nn.Embedding(MAX_SEGMENTS + 1, c_s)             # +1 SEP/pad slot
        self.relpos = Linear(2 * max_offset + 2, c_z)                  # +1 cross-segment bin
        self.left = Linear(c_s, c_z)
        self.right = Linear(c_s, c_z)
        self.anchor_state = Linear(N_ANCHOR_STATES, c_z)               # 3-way one-hot
        self.pe_emb = nn.Embedding(N_PE_BINS, c_z)                     # distance-bin PE

    def forward(self, tokens, residue_index, segment_id,
                anchor_state, pe_bin) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = segment_id.clamp(0, MAX_SEGMENTS)                        # SEP/pad -> last slot
        s0 = self.tok_emb(tokens) + self.seg_emb(seg)                  # [B,N,c_s]
        rel = self.relpos(_chain_relpos_onehot(residue_index, segment_id, self.max_offset))
        astate = self.anchor_state(
            F.one_hot(anchor_state.long().clamp(0, N_ANCHOR_STATES - 1),
                      num_classes=N_ANCHOR_STATES).float())
        z0 = (self.left(s0)[:, :, None, :] + self.right(s0)[:, None, :, :]
              + rel + astate + self.pe_emb(pe_bin.long().clamp(0, N_PE_BINS - 1)))
        return s0, z0


# --------------------------------------------------------------------------- #
# SingleUpdate (copied from src/model/utils.py — single attn biased by pair).
# --------------------------------------------------------------------------- #
class SingleUpdate(nn.Module):
    def __init__(self, d_s: int, d_z: int, no_heads: int, c_hidden: int,
                 transition_factor: int, inf: float = 1e9):
        super().__init__()
        self.inf = inf
        self.s_norm = LayerNorm(d_s)
        self.z_norm = LayerNorm(d_z)
        self.z_to_bias = Linear(d_z, no_heads, bias=False, init="normal")
        self.attn = Attention(d_s, d_s, d_s, c_hidden, no_heads, gating=True)
        self.transition = nn.Sequential(
            LayerNorm(d_s), Linear(d_s, d_s * transition_factor), nn.ReLU(),
            Linear(d_s * transition_factor, d_s),
        )

    def forward(self, s, z, seq_mask):
        s_n = self.s_norm(s)
        pair_bias = self.z_to_bias(self.z_norm(z)).permute(0, 3, 1, 2)   # [B,H,N,N]
        mask_bias = self.inf * (seq_mask[:, None, None, :] - 1.0)        # [B,1,1,N]
        s = s + self.attn(s_n, s_n, biases=[mask_bias, pair_bias])
        s = s + self.transition(s)
        return s


# --------------------------------------------------------------------------- #
# One single-row Evoformer-style block (the user's described data-flow).
# --------------------------------------------------------------------------- #
class SingleRowEvoBlock(nn.Module):
    """pair tri-mul + tri-attn -> single<-pair -> pair rowwise attn + transition
    -> pair<-single (outer product mean) -> pair transition."""

    def __init__(self, c_s: int, c_z: int, *, c_hidden_mul: Optional[int] = None,
                 c_hidden_tri: int = 32, no_heads_tri: int = 4, no_heads_s: int = 4,
                 c_hidden_s: Optional[int] = None, c_hidden_opm: int = 32,
                 transition_factor: int = 4, dropout: float = 0.1,
                 both_tri_mul: bool = True, pair_fp32: bool = False):
        super().__init__()
        # hidden widths scale with the (narrow) channel so the dominant O(N^3)
        # triangle-mul cost shrinks with c_z; tri-attn/opm heads stay small.
        c_hidden_mul = c_hidden_mul or c_z
        c_hidden_s = c_hidden_s or max(8, c_s // no_heads_s)
        # pair_fp32=True forces the pair stack to fp32. Only needed under *fp16*
        # autocast, where OpenFold's 1e9 inf-masking overflows (-> NaN on padded
        # rows). Under bf16 (our default) 1e9 is representable, so we leave it in
        # bf16 -> the O(N^3) triangle ops use tensor cores (the main speedup).
        self.pair_fp32 = pair_fp32
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = (TriangleMultiplicationIncoming(c_z, c_hidden_mul)
                           if both_tri_mul else None)
        self.tri_attn_start = TriangleAttentionStartingNode(c_z, c_hidden_tri, no_heads_tri)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z, c_hidden_tri, no_heads_tri)
        self.single = SingleUpdate(c_s, c_z, no_heads_s, c_hidden_s, transition_factor)
        self.pair_trans1 = PairTransition(c_z, transition_factor)
        self.opm = OuterProductMean(c_s, c_z, c_hidden_opm)
        self.pair_trans2 = PairTransition(c_z, transition_factor)
        self.drop = nn.Dropout(dropout)

    def _pair_ctx(self, dev: str):
        return (torch.autocast(device_type=dev, enabled=False)
                if self.pair_fp32 else contextlib.nullcontext())

    def forward(self, s, z, seq_mask, pair_mask):
        dev = z.device.type
        with self._pair_ctx(dev):
            if self.pair_fp32:
                z = z.float()
            z = z + self.drop(self.tri_mul_out(z, mask=pair_mask))
            if self.tri_mul_in is not None:
                z = z + self.drop(self.tri_mul_in(z, mask=pair_mask))
            z = z + self.drop(self.tri_attn_start(z, mask=pair_mask))
            z = z * pair_mask[..., None]
        s = self.single(s, z, seq_mask)                       # single <- pair
        s = s * seq_mask[..., None]
        with self._pair_ctx(dev):
            if self.pair_fp32:
                z = z.float()
            z = z + self.drop(self.tri_attn_end(z, mask=pair_mask))
            z = z + self.pair_trans1(z, mask=pair_mask)
            m_in = s.float() if self.pair_fp32 else s
            z = z + self.opm(m_in.unsqueeze(1), mask=seq_mask.float().unsqueeze(1))
            z = z + self.pair_trans2(z, mask=pair_mask)        # pair <- single
            z = z * pair_mask[..., None]
        return s, z


# --------------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------------- #
class PlddtModel(nn.Module):
    def __init__(self, c_s: int = 64, c_z: int = 64, n_blocks: int = 1,
                 dropout: float = 0.1, max_offset: int = 32,
                 both_tri_mul: bool = True, no_plddt_bins: int = 50,
                 pair_fp32: bool = False, grad_checkpoint: bool = False,
                 plddt_c: int = PLDDT_HEAD_DIM):
        super().__init__()
        self.c_s, self.c_z, self.plddt_c = c_s, c_z, plddt_c
        self.grad_checkpoint = grad_checkpoint
        self.featurizer = Featurizer(c_s, c_z, max_offset=max_offset)
        self.blocks = nn.ModuleList([
            SingleRowEvoBlock(c_s, c_z, dropout=dropout, both_tri_mul=both_tri_mul,
                              pair_fp32=pair_fp32)
            for _ in range(n_blocks)
        ])
        # Up-projection transition: the narrow single (c_s) is widened to the AF
        # pLDDT head's native dim so its pretrained weights load unchanged. This
        # adapter is always trainable (it learns to feed the frozen AF head).
        # NB: a normal (non-zero "final") init on the last layer so the adapter
        # passes signal into the pretrained AF head from step 0.
        self.plddt_proj = nn.Sequential(
            LayerNorm(c_s), Linear(c_s, plddt_c, init="relu"), nn.ReLU(),
            Linear(plddt_c, plddt_c, init="default"))
        self.plddt_head = PerResidueLDDTCaPredictor(
            no_bins=no_plddt_bins, c_in=plddt_c, c_hidden=128)
        self.mlm_head = Linear(c_s, VOCAB_SIZE, init="final")
        self.anchor_head = Linear(c_z, 1, init="final")               # binary anchor
        self.disto_head = Linear(c_z, N_DIST_BINS, init="final")      # distogram (B2)

    # -- representation + heads ------------------------------------------- #
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_mask = batch["token_mask"].float()                        # [B,N] incl SEP
        s, z = self.featurizer(batch["tokens"], batch["residue_index"],
                               batch["segment_id"], batch["anchor_state"],
                               batch["pe_bin"])
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                s, z = torch.utils.checkpoint.checkpoint(
                    blk, s, z, seq_mask, pair_mask, use_reentrant=False)
            else:
                s, z = blk(s, z, seq_mask, pair_mask)
        z_sym = 0.5 * (z + z.transpose(1, 2))
        return {
            "plddt_logits": self.plddt_head(self.plddt_proj(s)),      # up-proj -> AF head [B,N,50]
            "mlm_logits": self.mlm_head(s),                           # [B,N,VOCAB]
            "anchor_logits": self.anchor_head(z_sym).squeeze(-1),     # [B,N,N]
            "disto_logits": self.disto_head(z_sym),                   # [B,N,N,16]
            "single": s, "pair": z,
        }

    # -- weight init / freeze helpers ------------------------------------- #
    def load_af_plddt(self, path) -> None:
        sd = torch.load(path, weights_only=True)
        self.plddt_head.load_state_dict(sd)

    def set_plddt_trainable(self, flag: bool) -> None:
        for p in self.plddt_head.parameters():
            p.requires_grad_(flag)

    def plddt_head_groups(self):
        """Head sub-modules ordered output->input (gradual-unfreeze order)."""
        h = self.plddt_head
        return [("linear_3", h.linear_3), ("linear_2", h.linear_2),
                ("linear_1", h.linear_1), ("layer_norm", h.layer_norm)]

    def set_plddt_unfreeze(self, n_groups: int) -> List[str]:
        """Unfreeze the first ``n_groups`` head layers (from the output side);
        freeze the rest. n_groups=0 -> fully frozen, >=4 -> fully trainable.
        DDP-safe: the module is wrapped with all params trainable, so the reducer
        already tracks these; toggling requires_grad later syncs correctly."""
        for p in self.plddt_head.parameters():
            p.requires_grad_(False)
        unfrozen = []
        for name, mod in self.plddt_head_groups()[:max(0, n_groups)]:
            for p in mod.parameters():
                p.requires_grad_(True)
            unfrozen.append(name)
        return unfrozen

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        # full state dict (model4 trains everything; freeze only gates grads)
        return self.state_dict()


def predicted_plddt(plddt_logits: torch.Tensor) -> torch.Tensor:
    """Expected pLDDT in [0,100] from 50-bin logits (OpenFold convention)."""
    return compute_plddt(plddt_logits)


_DATAFLOW = r"""
 inputs: tokens[B,N]  residue_index[B,N]  segment_id[B,N]
         anchor_state[B,N,N]  pe_bin[B,N,N]   (token_mask[B,N])
   |
   v  Featurizer
   single s[B,N,{cs}]                pair z[B,N,N,{cz}]
   = tok_emb + seg_emb              = left(s)+right(s)
                                      + chain_relpos (per-segment + cross-seg bin)
                                      + anchor_state(3) + pe_bin(16)
   |
   |  x{nb} SingleRowEvoBlock:
   |     z += TriangleMultiplicationOutgoing(z){tmi}
   |     z += TriangleAttentionStartingNode(z)
   |     s  = SingleUpdate(s, z)        # single <- pair (attn bias) + transition
   |     z += TriangleAttentionEndingNode(z)
   |     z += PairTransition(z)
   |     z += OuterProductMean(s)       # pair  <- single
   |     z += PairTransition(z)
   v
 heads:  pLDDT( up-proj s:{cs}->{pc} -> AF head )->[B,N,50]   MLM(s)->[B,N,{vocab}]
         Anchor(sym z)->[B,N,N]   Distogram(sym z)->[B,N,N,16]
"""


def summarize_model(model: "PlddtModel", title: str = "model4 / PlddtModel") -> str:
    blk = model.blocks[0] if len(model.blocks) else None
    diagram = _DATAFLOW.format(
        cs=model.c_s, cz=model.c_z, pc=model.plddt_c, nb=len(model.blocks),
        vocab=VOCAB_SIZE,
        tmi=" + TriangleMultiplicationIncoming(z)"
            if (blk is not None and blk.tri_mul_in is not None) else "")
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = ["=" * 72, title, "=" * 72, diagram, "-" * 72,
             f"pair stack dtype : {'fp32 (forced)' if (blk and blk.pair_fp32) else 'autocast (bf16)'}",
             f"both tri-mul     : {blk.tri_mul_in is not None if blk else False}",
             f"grad checkpoint  : {model.grad_checkpoint}",
             f"parameters       : total={tot:,}  trainable={tr:,}", "-" * 72,
             f"{'module':22s}{'params':>14s}   class"]
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        lines.append(f"{name:22s}{n:>14,}   {mod.__class__.__name__}")
    lines.append("=" * 72)
    return "\n".join(lines)
