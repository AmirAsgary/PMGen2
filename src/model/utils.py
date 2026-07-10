"""
PMGen-v2 distillation — model + training utilities (the bulk lives here).

This module holds the encoder (7 pair-update variants behind one class), the
DistillModel wrapper around the FROZEN AF2 stack, and (added in PART 2) the
dataset, split builder, losses and train/eval helpers. ``train.py`` and the
``*_test.py`` gate scripts import from here.

Reused, never reimplemented:
  - load_frozen_fold (src/afbuild/utils.py): frozen AF2 structure module + heads.
  - parse_example / collate_fn (src/pdb/parse.py): teacher tensors + batching.
  - OpenFold blocks: Linear, LayerNorm, Attention, Triangle{Multiplication,
    Attention}{Outgoing/Incoming, StartingNode/EndingNode}; residue_constants.

Env: pmgen2  (~/miniforge3/envs/pmgen2/bin/python).
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm.auto import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Repo wiring: locate root, expose OpenFold, and load the validated helpers
# without name-clashing on the generic module name "utils".
# --------------------------------------------------------------------------- #
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "openfold").is_dir() and (parent / "src" / "afbuild").is_dir():
            return parent
    raise RuntimeError("could not locate PMGen2 repo root (openfold + src/afbuild)")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT / "openfold") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "openfold"))

from openfold.np import residue_constants as rc                       # noqa: E402
from openfold.config import model_config                              # noqa: E402
from openfold.model.primitives import Linear, LayerNorm, Attention    # noqa: E402
from openfold.model.triangular_multiplicative_update import (         # noqa: E402
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.model.triangular_attention import (                     # noqa: E402
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.data.data_transforms import (                           # noqa: E402
    atom37_to_frames,
    get_backbone_frames,
)
from openfold.utils.loss import (                                     # noqa: E402
    backbone_loss, compute_fape, sidechain_loss, supervised_chi_loss,
    compute_renamed_ground_truth)
from openfold.utils.rigid_utils import Rigid, Rotation                # noqa: E402


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_afbuild = _load_module("afbuild_utils", "src/afbuild/utils.py")
_pdbparse = _load_module("pdb_parse", "src/pdb/parse.py")
load_frozen_fold = _afbuild.load_frozen_fold
parse_example = _pdbparse.parse_example
collate_fn = _pdbparse.collate_fn

N_AATYPE: int = rc.restype_num + 1     # 21 (20 standard + unknown)
N_SEGMENTS: int = 3                    # 0=MHC/alpha, 1=beta/peptide, 2=peptide
_SEG_PAD: int = N_SEGMENTS             # embedding index reserved for padding (-1)
FROZEN_C_S: int = 384                  # fixed output widths the frozen stack wants
FROZEN_C_Z: int = 128


def set_seed(seed: int) -> None:
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Featurizer: tokens + AF2-style relative position -> (s0, z0)
# --------------------------------------------------------------------------- #
def _relpos_onehot(residue_index: torch.Tensor, max_offset: int) -> torch.Tensor:
    """AF2 relative-position one-hot, offset clipped to +-max_offset so the
    ~200 MHC/peptide numbering gap simply saturates to 'far'. -> [B,N,N,2k+1]."""
    diff = residue_index[:, :, None] - residue_index[:, None, :]       # [B,N,N]
    binned = diff.clamp(-max_offset, max_offset) + max_offset          # 0..2k
    return F.one_hot(binned, num_classes=2 * max_offset + 1).float()


class Featurizer(nn.Module):
    """Embeds aatype/segment/anchor into s0 and builds z0 from the outer
    combination of single tokens plus relative-position encoding."""

    def __init__(self, d_s: int, d_z: int, max_offset: int = 32) -> None:
        super().__init__()
        self.max_offset = max_offset
        self.aatype_emb = nn.Embedding(N_AATYPE, d_s)
        self.segment_emb = nn.Embedding(N_SEGMENTS + 1, d_s)   # +1 pad slot
        self.anchor_emb = nn.Embedding(2, d_s)
        self.relpos = Linear(2 * max_offset + 1, d_z)
        self.left = Linear(d_s, d_z)
        self.right = Linear(d_s, d_z)

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor,
                anchor: torch.Tensor, segment_id: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = torch.where(segment_id < 0,
                          torch.full_like(segment_id, _SEG_PAD), segment_id)
        s0 = (self.aatype_emb(aatype)
              + self.segment_emb(seg)
              + self.anchor_emb(anchor.long()))                       # [B,N,d_s]
        rel = self.relpos(_relpos_onehot(residue_index, self.max_offset))
        z0 = self.left(s0)[:, :, None, :] + self.right(s0)[:, None, :, :] + rel
        return s0, z0


# --------------------------------------------------------------------------- #
# Pair update (variant-selected) and single update
# --------------------------------------------------------------------------- #
_VARIANTS: Dict[int, List[str]] = {
    1: ["tri_mul_in", "tri_mul_out"],
    2: ["tri_attn_start", "tri_attn_end"],
    3: ["tri_mul_in"],
    4: ["tri_mul_out"],
    5: ["tri_attn_start"],
    6: ["tri_attn_end"],
    7: ["tri_mul_out", "tri_mul_in", "tri_attn_start", "tri_attn_end"],
}


class PairUpdate(nn.Module):
    """Variant-specific pair stack. Each op carries its own LayerNorm; we apply
    them with an external residual. Variant 7 also appends a pair transition."""

    def __init__(self, variant: int, d_z: int, c_hidden_mul: int,
                 c_hidden_tri: int, no_heads_tri: int, transition_factor: int):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be in 1..7, got {variant}")
        self.variant = variant
        builders = {
            "tri_mul_out": lambda: TriangleMultiplicationOutgoing(d_z, c_hidden_mul),
            "tri_mul_in": lambda: TriangleMultiplicationIncoming(d_z, c_hidden_mul),
            "tri_attn_start": lambda: TriangleAttentionStartingNode(
                d_z, c_hidden_tri, no_heads_tri),
            "tri_attn_end": lambda: TriangleAttentionEndingNode(
                d_z, c_hidden_tri, no_heads_tri),
        }
        self.ops = nn.ModuleList([builders[name]() for name in _VARIANTS[variant]])
        self.transition = None
        if variant == 7:
            self.transition = nn.Sequential(
                LayerNorm(d_z), Linear(d_z, d_z * transition_factor), nn.ReLU(),
                Linear(d_z * transition_factor, d_z),
            )

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        # Run the pair stack in fp32. OpenFold's TriangleAttention does NOT guard
        # autocast, so under --amp it runs in fp16 where its inf-masking (1e9)
        # overflows -> fully-padded rows softmax to NaN. fp32 keeps -1e9 finite.
        # (The tri-mul ops already force fp32 internally, so this is a no-op for
        # the mul-only variants.)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            for op in self.ops:                # uniform (z, mask=...) signature
                z = z + op(z, mask=pair_mask)
            if self.transition is not None:
                z = z + self.transition(z)
        return z


class SingleUpdate(nn.Module):
    """Self-attention on s biased by (a projection of) z, then a transition."""

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

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                seq_mask: torch.Tensor) -> torch.Tensor:
        s_n = self.s_norm(s)
        pair_bias = self.z_to_bias(self.z_norm(z)).permute(0, 3, 1, 2)   # [B,H,N,N]
        mask_bias = self.inf * (seq_mask[:, None, None, :] - 1.0)        # [B,1,1,N]
        s = s + self.attn(s_n, s_n, biases=[mask_bias, pair_bias])
        s = s + self.transition(s)
        return s


class EncoderBlock(nn.Module):
    """One pairformer-style block: update z FIRST, then update s with that z."""

    def __init__(self, variant: int, d_s: int, d_z: int, *, c_hidden_mul: int,
                 c_hidden_tri: int, no_heads_tri: int, no_heads_s: int,
                 c_hidden_s: int, transition_factor: int):
        super().__init__()
        self.pair = PairUpdate(variant, d_z, c_hidden_mul, c_hidden_tri,
                               no_heads_tri, transition_factor)
        self.single = SingleUpdate(d_s, d_z, no_heads_s, c_hidden_s,
                                   transition_factor)

    def forward(self, s: torch.Tensor, z: torch.Tensor, seq_mask: torch.Tensor,
                pair_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pair(z, pair_mask)
        z = z * pair_mask[..., None]
        s = self.single(s, z, seq_mask)
        s = s * seq_mask[..., None]
        return s, z


# --------------------------------------------------------------------------- #
# Encoder (7 variants) and DistillModel
# --------------------------------------------------------------------------- #
class RecyclingEmbedder(nn.Module):
    """AF2-style recycling: fold the previous iteration's outputs back into the
    initial single/pair. Adds LayerNorm(prev_s), LayerNorm(prev_z) and a binned
    Cα–Cα distogram of the previous structure (all init so it starts ~no-op)."""

    def __init__(self, d_s: int, d_z: int, min_bin: float = 3.25,
                 max_bin: float = 20.75, no_bins: int = 15):
        super().__init__()
        self.s_norm = LayerNorm(d_s)
        self.z_norm = LayerNorm(d_z)
        self.register_buffer("breaks", torch.linspace(min_bin, max_bin, no_bins - 1))
        self.linear_dist = Linear(no_bins, d_z, init="final")   # 0-init: no-op start

    def forward(self, prev_s, prev_z, prev_ca):
        d = torch.cdist(prev_ca, prev_ca)                        # [B,N,N]
        oh = F.one_hot(torch.bucketize(d, self.breaks),
                       num_classes=self.breaks.numel() + 1).to(prev_z.dtype)
        return self.s_norm(prev_s), self.z_norm(prev_z) + self.linear_dist(oh)


class DistillEncoder(nn.Module):
    """Stripped pairformer encoder: featurize -> ``depth`` blocks -> project to
    the frozen stack's exact widths s:[B,N,384], z:[B,N,N,128]. With ``recycle``,
    the previous iteration's (single, pair, Cα) are embedded back into s0/z0."""

    def __init__(self, variant: int, d_s: int = 384, d_z: int = 128,
                 depth: int = 1, *, c_hidden_mul: int = 128, c_hidden_tri: int = 32,
                 no_heads_tri: int = 4, no_heads_s: int = 8, c_hidden_s: int = 32,
                 transition_factor: int = 2, max_offset: int = 32,
                 recycle: bool = False):
        super().__init__()
        self.variant = variant
        self.featurizer = Featurizer(d_s, d_z, max_offset)
        self.blocks = nn.ModuleList([
            EncoderBlock(variant, d_s, d_z, c_hidden_mul=c_hidden_mul,
                         c_hidden_tri=c_hidden_tri, no_heads_tri=no_heads_tri,
                         no_heads_s=no_heads_s, c_hidden_s=c_hidden_s,
                         transition_factor=transition_factor)
            for _ in range(depth)
        ])
        self.s_out_norm = LayerNorm(d_s)
        self.z_out_norm = LayerNorm(d_z)
        # default (non-zero) init: these are the PRIMARY outputs feeding the
        # frozen stack, not residual branches — zero-init ('final') would start
        # the SM from a black hole with no gradient signal.
        self.s_out = Linear(d_s, FROZEN_C_S)
        self.z_out = Linear(d_z, FROZEN_C_Z)
        # recycler embeds the SM's c_s single + our c_z pair from the prev pass
        self.recycler = RecyclingEmbedder(FROZEN_C_S, d_z) if recycle else None

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor,
                seq_mask: torch.Tensor, anchor: torch.Tensor,
                segment_id: torch.Tensor, recyc=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]          # [B,N,N]
        s, z = self.featurizer(aatype, residue_index, anchor, segment_id)
        if self.recycler is not None and recyc is not None:
            s_up, z_up = self.recycler(*recyc)                  # prev (s, z, ca)
            s = s + s_up
            z = z + z_up
        s = s * seq_mask[..., None]
        z = z * pair_mask[..., None]
        for block in self.blocks:
            s, z = block(s, z, seq_mask, pair_mask)
        s_out = self.s_out(self.s_out_norm(s)) * seq_mask[..., None]
        z_out = self.z_out(self.z_out_norm(z)) * pair_mask[..., None]
        return s_out, z_out


def anchor_canonical_resindex(residue_index, anchor, seq_mask, segment_id,
                              span: int = 8):
    """Re-number the PEPTIDE so the alignment register shows up in relpos (and thus
    in z, the only channel into the frozen SM). Anchors are pinned to canonical
    groove slots: the inter-anchor residues are left-aligned, leaving a NUMBERING
    GAP (the bulge) before the C-terminal anchor when the peptide is shorter than
    the canonical span. Different anchor pairs -> different relpos -> the SM can
    tell registers apart positionally (the same signal AF2 got from the gapped
    template). MHC numbering and the ~200 MHC->peptide gap are preserved.

    Derived purely from the anchor flag already in the batch (no re-preprocess).
    EXPERIMENTAL / heuristic: the faithful alternative is PMGen's exact gap string.
    """
    out = residue_index.clone()
    B = residue_index.shape[0]
    for b in range(B):
        m = seq_mask[b].bool()
        if not m.any():
            continue
        seg = segment_id[b]
        pep = (seg == seg[m].max()) & m                  # peptide = highest segment
        pep_pos = pep.nonzero(as_tuple=False).flatten()
        L = int(pep_pos.numel())
        if L == 0:
            continue
        mhc = m & ~pep
        base = (int(residue_index[b][mhc].max()) if mhc.any() else 0) + 200
        anc = (anchor[b][pep_pos] > 0.5).nonzero(as_tuple=False).flatten().tolist()
        if len(anc) < 2:                                 # 0/1 anchor -> contiguous
            slots = list(range(L))
        else:
            af, al = anc[0], anc[-1]
            c_slot = max(span, (al - af))                # gap before C-anchor if short
            slots = [(k - af) if k < al else
                     (c_slot if k == al else c_slot + (k - al)) for k in range(L)]
        s0 = slots[0]
        for k in range(L):
            out[b, pep_pos[k]] = base + (slots[k] - s0)
    return out


class DistillModel(nn.Module):
    """Trainable encoder + FROZEN AF2 stack. Only the encoder has gradients."""

    def __init__(self, variant: int, model_name: str = "model_2_ptm",
                 device: str = "cpu", recycles: int = 0,
                 unfreeze_sm: float = 0.0, unfreeze_plddt: float = 0.0,
                 unfreeze_pae: float = 0.0, anchor_relpos: bool = False,
                 **encoder_kwargs):
        super().__init__()
        self.recycles = int(recycles)
        self.anchor_relpos = bool(anchor_relpos)
        self.encoder = DistillEncoder(variant, recycle=self.recycles > 0,
                                      **encoder_kwargs)
        self.frozen = load_frozen_fold(model_name, device=device)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        # optionally make the LAST pct% of each frozen sub-module trainable
        self.unfrozen = {
            "sm": self._unfreeze_last_pct(self.frozen.sm, unfreeze_sm),
            "plddt": self._unfreeze_last_pct(self.frozen.plddt, unfreeze_plddt),
            "pae": self._unfreeze_last_pct(self.frozen.tm, unfreeze_pae),
        }
        self.to(device)

    @staticmethod
    def _unfreeze_last_pct(module: nn.Module, pct: float) -> float:
        """Set requires_grad=True on whole parameter tensors from the END of the
        module until ~pct% of its parameters (by count) are trainable. Returns the
        actual fraction unfrozen. pct<=0 -> nothing (stays fully frozen)."""
        if pct is None or pct <= 0:
            return 0.0
        named = list(module.named_parameters())
        total = sum(p.numel() for _, p in named) or 1
        target, acc = (pct / 100.0) * total, 0
        for _, p in reversed(named):
            if acc >= target:
                break
            p.requires_grad_(True)
            acc += p.numel()
        return acc / total

    def train(self, mode: bool = True) -> "DistillModel":
        super().train(mode)
        self.frozen.eval()                     # eval mode (no dropout/BN updates)
        return self                            #   even for the unfrozen params

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def frozen_trainable_state(self) -> Dict[str, torch.Tensor]:
        """state_dict of the unfrozen FROZEN-stack params (empty when 0% unfrozen);
        saved/loaded separately so the bulk frozen weights need not be stored."""
        return {n: p.detach().cpu() for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("encoder.")}

    def forward(self, batch: Dict[str, torch.Tensor], return_frames: bool = False,
                num_recycles: Optional[int] = None):
        """-> (ca[B,N,3], plddt_logits[B,N,50], pae_logits[B,N,N,64]) and, if
        ``return_frames``, the SM backbone-frame trajectory for FAPE.

        Recycling: runs ``num_recycles``+1 trunk passes, feeding the previous
        (SM single, pair, Cα) back through the encoder. Earlier passes run under
        no_grad (AF2-style) so only the last pass is backpropped — extra passes
        cost forward-only. ``num_recycles`` defaults to ``self.recycles``."""
        nr = self.recycles if num_recycles is None else num_recycles
        if self.encoder.recycler is None:
            nr = 0
        aa, ri = batch["aatype"], batch["residue_index"]
        sm_, an, seg = batch["seq_mask"], batch["anchor"], batch["segment_id"]
        if self.anchor_relpos:                           # register -> relpos -> z
            ri = anchor_canonical_resindex(ri, an, sm_, seg)

        recyc = None
        out = z = ca = None
        for it in range(nr + 1):
            is_last = it == nr
            with (nullcontext() if is_last else torch.no_grad()):
                s, z = self.encoder(aa, ri, sm_, an, seg, recyc=recyc)
                out = self.frozen.sm({"single": s, "pair": z}, aa, mask=sm_)
                ca = out["positions"][-1][..., 1, :]
                if not is_last:
                    recyc = (out["single"].detach(), z.detach(), ca.detach())
        plddt_logits = self.frozen.plddt(out["single"])
        pae_logits = self.frozen.tm(z)
        if return_frames:
            return ca, plddt_logits, pae_logits, out["frames"]
        return ca, plddt_logits, pae_logits


# --------------------------------------------------------------------------- #
# Small helpers shared by the gate scripts
# --------------------------------------------------------------------------- #
DUMMY_DIR = REPO_ROOT / "data" / "test"


def load_dummy_examples(ids: List[str] | None = None, limit: int | None = None
                        ) -> List[Dict[str, torch.Tensor]]:
    """Parse example(s) from data/test/ (the --dummy source) via the validated
    parser. Returns a list of per-example tensor dicts (true length, no pad)."""
    import csv
    with open(DUMMY_DIR / "inputs.tsv") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    chosen = ids if ids is not None else sorted(rows)
    if limit is not None:
        chosen = chosen[:limit]
    out = []
    for fid in chosen:
        r = rows[fid]
        pdb = DUMMY_DIR / "pdbs" / "alphafold" / fid / f"{fid}_model_1_model_2_ptm.pdb"
        out.append(parse_example(pdb, r["peptide"], r["mhc_seq"],
                                 r["anchors"], r["mhc_type"]))
    return out


# =========================================================================== #
# PART 2 — data: teacher loading, Dataset, collate extension, split builder
# =========================================================================== #
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
POOL_CSV = PROCESSED_DIR / "full_dataset_pmgeninput.csv"
# slim, repo-friendly ordered base-id list (just the 'id' column of POOL_CSV);
# read_split_ids prefers it so the 70 MB pool CSV need not be pushed.
POOL_IDS_CSV = PROCESSED_DIR / "base_ids_ordered.csv"
# base ids with >=1 existing PMGen structure (written by the post-prediction
# cleaning script); when present, read_split_ids intersects every split with it.
VALID_BASE_IDS_CSV = PROCESSED_DIR / "valid_base_ids.csv"
DUMMY_AF_DIR = DUMMY_DIR / "pdbs" / "alphafold"


def find_teacher_files(af_dir: Path) -> Tuple[Path, Path, Path]:
    """Locate (pdb, plddt.npy, pae.npy) inside a PMGen alphafold output dir.

    PAE is the *_predicted_aligned_error.npy file (NOT *_ptm.npy, which is the
    scalar pTM). Raises if any is missing or ambiguous.
    """
    af_dir = Path(af_dir)

    def _one(pattern: str, what: str) -> Path:
        hits = sorted(glob.glob(str(af_dir / pattern)))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{af_dir}: expected exactly one {what} matching '{pattern}', "
                f"found {len(hits)}")
        return Path(hits[0])

    pdb = _one("*_model_1_model_2_ptm.pdb", "teacher PDB")
    plddt = _one("*_plddt.npy", "pLDDT npy")
    pae = _one("*_predicted_aligned_error.npy", "PAE npy")
    return pdb, plddt, pae


def sidechain_gt_from_atom14(aatype: torch.Tensor, atom14: torch.Tensor,
                             atom14_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """AF2's side-chain ground-truth features, derived (batched, on-device) from the
    stored ``teacher_atom14`` — a lossless, compact repacking of atom37.

    Produces exactly the tensors ``openfold.utils.loss.sidechain_loss`` and
    ``compute_renamed_ground_truth`` consume:
      rigidgroups_{gt_frames, alt_gt_frames, gt_exists}   -- the 8 rigid groups/residue
      atom14_{gt,alt_gt}_{positions,exists}, atom14_atom_is_ambiguous
    The alt/ambiguous tensors are what let AF2 score symmetric side chains (Asp OD1/OD2,
    Phe CD1/CD2, ...) without punishing an equivalent atom naming.
    """
    from openfold.data.data_transforms import (                      # noqa: E402
        make_atom14_masks, make_atom14_positions, atom37_to_frames)
    from openfold.utils.feats import atom14_to_atom37                # noqa: E402
    from openfold.utils.tensor_utils import batched_gather           # noqa: E402

    p: Dict[str, torch.Tensor] = make_atom14_masks({"aatype": aatype})
    a37 = atom14_to_atom37(atom14, p)                                # [*, N, 37, 3]
    nb = len(atom14_mask.shape[:-1])
    m37 = batched_gather(atom14_mask, p["residx_atom37_to_atom14"],
                         dim=-1, no_batch_dims=nb) * p["atom37_atom_exists"]
    prot = {"aatype": aatype, "all_atom_positions": a37, "all_atom_mask": m37, **p}
    prot = atom37_to_frames(prot)
    prot = make_atom14_positions(prot)
    keys = ("rigidgroups_gt_frames", "rigidgroups_alt_gt_frames", "rigidgroups_gt_exists",
            "atom14_gt_positions", "atom14_alt_gt_positions", "atom14_gt_exists",
            "atom14_alt_gt_exists", "atom14_atom_is_ambiguous")
    return {k: prot[k] for k in keys}


def load_teacher_arrays(plddt_npy: Path, pae_npy: Path, n: int
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load + slice the (post-padded) teacher arrays to the true length N.

    plddt -> [N] per-residue (0..100); pae -> [N,N] Ångström. Validates that the
    stored arrays are at least N (padding may make them longer)."""
    plddt = np.load(plddt_npy)
    pae = np.load(pae_npy)
    if plddt.ndim != 1 or plddt.shape[0] < n:
        raise ValueError(f"{plddt_npy}: plddt shape {plddt.shape}, need >= ({n},)")
    if pae.ndim != 2 or pae.shape[0] < n or pae.shape[1] < n:
        raise ValueError(f"{pae_npy}: pae shape {pae.shape}, need >= ({n},{n})")
    plddt_t = torch.from_numpy(np.ascontiguousarray(plddt[:n])).float()
    pae_t = torch.from_numpy(np.ascontiguousarray(pae[:n, :n])).float()
    return plddt_t, pae_t


class DistillDataset(Dataset):
    """One item per row -> parse_example(...) + sliced teacher pLDDT/PAE.

    ``rows`` are dicts with keys: id, peptide, mhc_seq, anchors, mhc_type,
    alphafold_output_path (a PMGen output dir for that id).
    """

    def __init__(self, rows: Sequence[Dict[str, str]]):
        if not rows:
            raise ValueError("DistillDataset received no rows")
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        pdb, plddt_npy, pae_npy = find_teacher_files(Path(r["alphafold_output_path"]))
        ex = parse_example(pdb, r["peptide"], r["mhc_seq"], r["anchors"],
                           r["mhc_type"], return_backbone=True)
        n = int(ex["aatype"].shape[0])
        ex["teacher_plddt"], ex["teacher_pae"] = load_teacher_arrays(
            plddt_npy, pae_npy, n)
        ex["id"] = r["id"]
        return ex


def collate_with_teacher(examples: Sequence[Dict[str, torch.Tensor]]
                         ) -> Dict[str, torch.Tensor]:
    """Reuse the validated ``collate_fn`` for the parsed tensors, then pad the
    teacher pLDDT [B,N] and PAE [B,N,N] (padding = 0; masked downstream)."""
    batch = collate_fn(examples)                      # base parsed + pad
    b, max_n = batch["aatype"].shape
    teacher_plddt = torch.zeros(b, max_n, dtype=torch.float32)
    teacher_pae = torch.zeros(b, max_n, max_n, dtype=torch.float32)
    has_bb = "teacher_bb" in examples[0]
    teacher_bb = torch.zeros(b, max_n, 3, 3, dtype=torch.float32) if has_bb else None
    has_chi = "teacher_chi" in examples[0]
    teacher_chi = torch.zeros(b, max_n, 4, 2, dtype=torch.float32) if has_chi else None
    teacher_chi_mask = torch.zeros(b, max_n, 4, dtype=torch.float32) if has_chi else None
    for i, e in enumerate(examples):
        n = int(e["aatype"].shape[0])
        teacher_plddt[i, :n] = e["teacher_plddt"]
        teacher_pae[i, :n, :n] = e["teacher_pae"]
        if has_bb:
            teacher_bb[i, :n] = e["teacher_bb"]
        if has_chi:
            teacher_chi[i, :n] = e["teacher_chi"]
            teacher_chi_mask[i, :n] = e["teacher_chi_mask"]
    batch["teacher_plddt"] = teacher_plddt
    batch["teacher_pae"] = teacher_pae
    if has_bb:
        batch["teacher_bb"] = teacher_bb
    if has_chi:
        batch["teacher_chi"] = teacher_chi
        batch["teacher_chi_mask"] = teacher_chi_mask
    if "id" in examples[0]:
        batch["id"] = [e["id"] for e in examples]
    # per-example loss weights (default 1.0 -> no-op). sample_weight scales FAPE+CE
    # (source down-weighting); struct_weight scales FAPE only (structure quality w_n).
    for key in ("sample_weight", "struct_weight"):
        if any(key in e for e in examples):
            batch[key] = torch.tensor(
                [float(e.get(key, 1.0)) for e in examples], dtype=torch.float32)
    # AF2 sidechain targets (side-chain stores only): pad with zeros; the *_mask
    # tensors mark the padding, and the rigid-group / renaming features are derived
    # from these on-device in the loss (see sidechain_gt_from_atom14).
    for key, tail in (("teacher_atom14", (14, 3)), ("teacher_atom14_mask", (14,)),
                      ("teacher_chi", (4, 2)), ("teacher_chi_mask", (4,))):
        if key in examples[0]:
            t = torch.zeros(b, max_n, *tail, dtype=torch.float32)
            for i, e in enumerate(examples):
                t[i, : int(e["aatype"].shape[0])] = e[key]
            batch[key] = t
    return batch


def base_id(anchor_id: str) -> str:
    """Strip the trailing '_<anchor_index>' from an anchor-expanded id to recover
    the per-(peptide,MHC) base id used by the splits (e.g.
    'BOLA100901_ELGNITGN_8_3' -> 'BOLA100901_ELGNITGN_8')."""
    return anchor_id.rsplit("_", 1)[0]


def _read_row_idx(path: Path) -> List[int]:
    with open(path) as fh:
        return [int(r["row_idx"]) for r in csv.DictReader(fh)]


def read_split_ids(scheme: str, fold: int,
                   pool_csv: Path = POOL_CSV) -> Dict[str, List[str]]:
    """Build {train,val,test} base-id lists for a scheme/fold from the processed
    split files. ``row_idx`` indexes into the pool CSV (same row order as
    full_dataset.csv). train = pool − this fold's val − test.

    If ``valid_base_ids.csv`` exists (written by
    src/post_structure_prediction_processing/script.py), every split is
    intersected with it, so base ids whose PMGen structures don't exist are
    excluded from train/val/test everywhere.
    """
    if scheme not in ("two_axis", "hla_only"):
        raise ValueError(f"scheme must be two_axis|hla_only, got {scheme!r}")
    if not (1 <= fold <= 5):
        raise ValueError(f"fold must be 1..5, got {fold}")
    import pandas as pd
    ids_src = POOL_IDS_CSV if POOL_IDS_CSV.exists() else pool_csv
    ids = pd.read_csv(ids_src, usecols=["id"], dtype=str)["id"].tolist()
    base = PROCESSED_DIR / scheme
    test_idx = _read_row_idx(base / "test" / "test.csv")
    val_idx = _read_row_idx(base / "cv" / f"fold_{fold}" / "val.csv")
    test_ids = {ids[i] for i in test_idx}
    val_ids = {ids[i] for i in val_idx}
    held = test_ids | val_ids
    train_ids = [i for i in ids if i not in held]
    result = {"train": train_ids, "val": sorted(val_ids), "test": sorted(test_ids)}
    if VALID_BASE_IDS_CSV.exists():
        valid = set(pd.read_csv(VALID_BASE_IDS_CSV, dtype=str)["id"])
        result = {k: [x for x in v if x in valid] for k, v in result.items()}
    return result


def dummy_rows() -> List[Dict[str, str]]:
    """Rows for --dummy mode: the 15 class-I examples in data/test/, with each
    id's alphafold_output_path pointing at its own output dir."""
    with open(DUMMY_DIR / "inputs.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["alphafold_output_path"] = str(DUMMY_AF_DIR / r["id"])
    return rows


ANCHORS_TSV = (REPO_ROOT / "outputs" / "pmgen_input" / "full_dataset"
               / "Multiple_Anchors_input_reduced.tsv")


def build_dataset(scheme: str = "two_axis", fold: int = 1, split: str = "train",
                  dummy: bool = False, pool_csv: Path = POOL_CSV,
                  anchors_tsv: Path = ANCHORS_TSV,
                  af_root: Optional[Path] = None) -> DistillDataset:
    """Construct a DistillDataset for a split.

    --dummy: the SAME code path over data/test/ (all 15 class-I examples,
    regardless of scheme/fold/split).

    Real mode: one training example per *anchor combination* (rows of
    ``anchors_tsv``), with each anchor row assigned to the split of its **base
    id** (``base_id`` strips the anchor suffix). This guarantees that *all*
    anchor combinations of a (peptide, MHC) pair land in the SAME split. Each
    anchor id's teacher PMGen output dir is ``af_root/<anchor_id>`` (af_root must
    be supplied once teacher predictions exist).
    """
    if dummy:
        return DistillDataset(dummy_rows())
    if af_root is None:
        raise RuntimeError(
            "real-mode build_dataset needs --af-root: the directory holding one "
            "PMGen teacher output sub-dir per anchor id. Use --dummy for local "
            "runs (teacher predictions for the full anchor pool don't exist yet).")

    import pandas as pd
    split_base = set(read_split_ids(scheme, fold, pool_csv)[split])
    anchors = pd.read_csv(anchors_tsv, sep="\t").to_dict("records")
    af_root = Path(af_root)
    rows = []
    for r in anchors:
        if base_id(str(r["id"])) not in split_base:
            continue
        r["alphafold_output_path"] = str(af_root / str(r["id"]))
        rows.append(r)
    return DistillDataset(rows)


# =========================================================================== #
# PART 2 — losses (3-term) and evaluation metrics
# =========================================================================== #
def _build_gt_backbone_frames(aatype: torch.Tensor, teacher_bb: torch.Tensor,
                              seq_mask: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher backbone N/Cα/C -> AF2 backbone rigid frames, reusing OpenFold's
    ``atom37_to_frames`` + ``get_backbone_frames`` (group 0). Returns
    (backbone_rigid_tensor [B,N,4,4], backbone_rigid_mask [B,N])."""
    b, n = aatype.shape
    atom37 = teacher_bb.new_zeros(b, n, rc.atom_type_num, 3)
    atom37[..., :3, :] = teacher_bb                       # N=0, CA=1, C=2
    mask37 = teacher_bb.new_zeros(b, n, rc.atom_type_num)
    mask37[..., :3] = seq_mask[..., None]
    protein = {"aatype": aatype, "all_atom_positions": atom37,
               "all_atom_mask": mask37}
    protein = get_backbone_frames(atom37_to_frames(protein))
    return protein["backbone_rigid_tensor"], protein["backbone_rigid_mask"]


def bin_plddt(plddt: torch.Tensor, no_bins: int) -> torch.Tensor:
    """Per-residue pLDDT (0..100) -> class index in [0, no_bins) (AF2 lddt bins
    uniformly over [0,100])."""
    width = 100.0 / no_bins
    return torch.clamp((plddt / width).floor().long(), 0, no_bins - 1)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    c = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, c), target.reshape(-1),
                         reduction="none").reshape(target.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def _backbone_fape_per_example(backbone_rigid_tensor, backbone_rigid_mask, traj,
                               B, pair_mask=None, clamp_distance=None,
                               loss_unit_distance=10.0, eps=1e-4):
    """Same FAPE as openfold ``backbone_loss`` but returns one value per example
    ([B]) instead of the batch-mean scalar, so the per-example loss can be
    reweighted (e.g. by teacher confidence) before reduction. ``traj`` carries
    leading trajectory-block dims; we average those out and keep the batch dim."""
    if traj.shape[-1] == 7:
        pred_aff = Rigid.from_tensor_7(traj)
    else:
        pred_aff = Rigid.from_tensor_4x4(traj)
    pred_aff = Rigid(Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(),
                              quats=None), pred_aff.get_trans())
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)
    fape = compute_fape(                                  # [*, B] (leading = blocks)
        pred_aff, gt_aff[None], backbone_rigid_mask[None],
        pred_aff.get_trans(), gt_aff[None].get_trans(), backbone_rigid_mask[None],
        pair_mask=pair_mask, l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance, eps=eps)
    return fape.reshape(-1, B).mean(0)                    # mean over blocks -> [B]


def _groove_terms(ca_teacher, ca_pred, pep, seq_mask, tau_mid=8.0, s_buried=1.5,
                  tau_out=8.0, tau_far=18.0):
    """Groove-membership geometry for the peptide, from per-residue nearest-MHC Cα
    distance (Å). Returns (buried[B,N], wout[B,N], contain):
      buried = soft 1-in-pocket / 0-outside (from the TEACHER) -> gates the exact
               structural (FAPE) loss so we only dock the in-pocket residues;
      wout   = (1-buried) on the peptide -> weights the containment;
      contain= soft-band containment on the PREDICTED structure: out-of-pocket
               residues are pushed into a [tau_out, tau_far] Å shell from the MHC
               (outside the groove, but not flown away). Direction-free."""
    INF = 1e9
    mhc_col = (seq_mask.bool() & ~pep.bool()).float()[:, None, :]       # [B,1,N]
    pepf = (pep.bool() & seq_mask.bool()).float()                      # [B,N]
    with torch.no_grad():                                              # label: no grad
        dt = torch.cdist(ca_teacher, ca_teacher) + (1.0 - mhc_col) * INF
        nnd_t = dt.min(-1).values                                     # teacher nearest-MHC
        bur_raw = torch.sigmoid((tau_mid - nnd_t) / s_buried)         # ~1 if close
    buried = bur_raw * pepf
    wout = (1.0 - bur_raw) * pepf
    dp = torch.cdist(ca_pred, ca_pred) + (1.0 - mhc_col) * INF        # predicted (grad)
    nnd_p = dp.min(-1).values
    band = F.relu(tau_out - nnd_p) ** 2 + F.relu(nnd_p - tau_far) ** 2
    contain = (wout * band).sum() / wout.sum().clamp_min(1.0)
    return buried, wout, contain


class DistillLoss(nn.Module):
    """L = λ_fape·FAPE + λ_plddt·CE(plddt, bin50) + λ_pae·CE(pae, bin64).

    The CE terms backprop through the (frozen, ungradiented) heads + SM into the
    encoder. Bin breakpoints come from the frozen model's loss config, not hard-
    coded: PAE no_bins/max_bin from ``loss.tm``; pLDDT no_bins from ``heads.lddt``.
    """

    def __init__(self, lambda_fape: float = 1.0, lambda_plddt: float = 0.1,
                 lambda_pae: float = 0.1, model_name: str = "model_2_ptm",
                 fape_clamp: Optional[float] = None, fape_unit: float = 10.0,
                 peptide_weight: float = 5.0, plddt_weight_struct: bool = False,
                 plddt_weight_floor: float = 0.1, groove_aware: bool = False,
                 lambda_contain: float = 0.5, groove=(8.0, 1.5, 8.0, 18.0),
                 lambda_sc_fape: float = 0.0, lambda_chi: float = 0.0,
                 chi_weight: float = 0.5, angle_norm_weight: float = 0.01,
                 sc_clamp_distance: float = 10.0, sc_length_scale: float = 10.0):
        # fape_clamp=None -> unclamped FAPE: needed to fold from a random init
        # (a 10 Å clamp zeroes the gradient when every error exceeds it). AF2's
        # clamped/clamp-schedule variant can be enabled once structures are close.
        # peptide_weight=1.0 -> uniform (identical to the unweighted loss); >1
        # up-weights peptide residues (and peptide-involving pairs) so the small
        # peptide is not drowned out by the large MHC.
        # plddt_weight_struct -> weight each example's FAPE by the teacher's median
        # peptide pLDDT (/100), so low-confidence teacher structures (whose Cα
        # targets are unreliable) contribute proportionally less to the geometry
        # loss. plddt_weight_floor keeps the weakest examples from vanishing.
        super().__init__()
        self.l_fape, self.l_plddt, self.l_pae = lambda_fape, lambda_plddt, lambda_pae
        self.fape_clamp, self.fape_unit = fape_clamp, fape_unit
        self.peptide_weight = float(peptide_weight)
        self.plddt_weight_struct = bool(plddt_weight_struct)
        self.plddt_weight_floor = float(plddt_weight_floor)
        # groove_aware: gate the exact FAPE by teacher groove-membership and add a
        # soft-band containment so out-of-pocket peptides are pushed outside (not
        # docked) — the model learns to DOCK binders and EXPEL non-binders, while
        # pLDDT (still trained on every peptide residue) flags the non-binders.
        self.groove_aware = bool(groove_aware)
        self.lambda_contain = float(lambda_contain)
        self.groove = tuple(groove)              # (tau_mid, s_buried, tau_out, tau_far)
        # AF2 side-chain supervision (Suppl. 1.9.x). Both default OFF so existing models
        # are bit-identical; they are enabled only when the model returns torsion `aux`
        # AND the store carries teacher sidechain targets. AF's own weights:
        #   fape.backbone 0.5, fape.sidechain 0.5, supervised_chi 1.0
        #   (chi_weight 0.5, angle_norm_weight 0.01)
        self.l_sc_fape = float(lambda_sc_fape)
        self.l_chi = float(lambda_chi)
        self.chi_weight = float(chi_weight)
        self.angle_norm_weight = float(angle_norm_weight)
        self.sc_clamp_distance = float(sc_clamp_distance)
        self.sc_length_scale = float(sc_length_scale)
        cfg = model_config(model_name)
        self.pae_no_bins = int(cfg["loss"]["tm"]["no_bins"])
        self.plddt_no_bins = int(cfg["model"]["heads"]["lddt"]["no_bins"])
        pae_max = float(cfg["loss"]["tm"]["max_bin"])
        self.register_buffer(
            "pae_breaks", torch.linspace(0.0, pae_max, self.pae_no_bins - 1))

    def forward(self, ca: torch.Tensor, plddt_logits: torch.Tensor,
                pae_logits: Optional[torch.Tensor], frames: torch.Tensor,
                batch: Dict[str, torch.Tensor],
                aux: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_mask = batch["seq_mask"]
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
        pep = peptide_mask_from_batch(seq_mask, batch["segment_id"])       # [B,N]
        pep_pair = torch.maximum(pep[:, :, None], pep[:, None, :])         # i or j peptide

        # per-residue / per-pair weights. w == 1.0 everywhere -> identical to the
        # unweighted loss (so existing checkpoints continue unchanged).
        w = self.peptide_weight
        if w != 1.0:
            res_w = seq_mask * (1.0 + (w - 1.0) * pep)                     # [B,N]
            pair_w = pair_mask * (1.0 + (w - 1.0) * pep_pair)             # [B,N,N]
            fape_pair = (1.0 + (w - 1.0) * pep)[:, None, :]               # weight pos j
        else:
            res_w, pair_w, fape_pair = seq_mask, pair_mask, None

        # per-example source weight [B] (e.g. down-weight low-diversity hasmig data);
        # folds into the CE weights (their ÷Σw normalization then down-weights the
        # example) and, below, into the FAPE reduction. Absent -> unchanged.
        sw = batch.get("sample_weight", None)
        if sw is not None:
            res_w = res_w * sw[:, None]
            pair_w = pair_w * sw[:, None, None]

        # groove-aware: replace the uniform peptide FAPE weight with a buried-gated
        # one (in-pocket peptide -> full structural weight; out-of-pocket -> ~0) and
        # compute the containment term. res_w/pair_w (pLDDT/PAE) are left covering all
        # peptide residues so confidence is still learned for the expelled ones.
        contain = ca.new_zeros(())
        buried_mean = ca.new_zeros(())
        if self.groove_aware:
            buried, wout, contain = _groove_terms(
                batch["teacher_ca"], ca, pep, seq_mask, *self.groove)
            mhc_valid = (seq_mask.bool() & ~pep.bool()).float()           # [B,N]
            fape_pair = (mhc_valid + w * buried)[:, None, :]              # weight pos j
            buried_mean = buried.sum() / pep.sum().clamp_min(1.0)

        bb_tensor, bb_mask = _build_gt_backbone_frames(
            batch["aatype"], batch["teacher_bb"], seq_mask)
        B = seq_mask.shape[0]

        def _fape_be(pmask):                              # per-example FAPE -> [B]
            return _backbone_fape_per_example(
                bb_tensor, bb_mask, frames, B, pair_mask=pmask,
                clamp_distance=self.fape_clamp, loss_unit_distance=self.fape_unit)

        fape_be = _fape_be(fape_pair)                     # [B]
        # Per-example FAPE weight = product of any enabled example weightings, then a
        # weighted mean (÷Σw keeps the FAPE scale stable). Three composable sources:
        #   plddt_weight_struct — teacher median peptide pLDDT/100 (floored)
        #   sample_weight (sw)  — source weight (also applied to the CE terms)
        #   struct_weight (stw) — precomputed structure-quality weight w_n, FAPE-ONLY
        #                         (so the pLDDT CE still learns confidence on ALL data)
        stw = batch.get("struct_weight", None)
        wfape, weighted = fape_be.new_ones(B), False
        if self.plddt_weight_struct:
            tp = batch["teacher_plddt"]
            g = fape_be.new_empty(B)
            for b in range(B):
                vals = tp[b][pep[b] > 0.5]
                g[b] = vals.median() if vals.numel() else tp.new_tensor(0.0)
            wfape = wfape * (g / 100.0).clamp(min=self.plddt_weight_floor, max=1.0)
            weighted = True
        if sw is not None:
            wfape, weighted = wfape * sw, True
        if stw is not None:
            wfape, weighted = wfape * stw, True
        fape = ((wfape * fape_be).sum() / wfape.sum().clamp_min(1e-4)
                if weighted else fape_be.mean())
        plddt_bins = bin_plddt(batch["teacher_plddt"], self.plddt_no_bins)
        ce_plddt = _masked_ce(plddt_logits, plddt_bins, res_w)
        # PAE is skipped entirely when lambda_pae == 0 (or the model returns None): the
        # [B,N,N,64] bucketize + CE is the single most expensive term and was multiplied
        # by zero. `pae_logits=None` is the model saying "no PAE head".
        if self.l_pae != 0.0 and pae_logits is not None:
            pae_bins = torch.bucketize(batch["teacher_pae"], self.pae_breaks)
            ce_pae = _masked_ce(pae_logits, pae_bins, pair_w)
        else:
            ce_pae = fape.new_zeros(())
        total = self.l_fape * fape + self.l_plddt * ce_plddt + self.l_pae * ce_pae
        if self.groove_aware:
            total = total + self.lambda_contain * contain

        # ---- AF2 side-chain supervision -------------------------------------------
        sc_fape = fape.new_zeros(())
        chi_loss = fape.new_zeros(())
        if aux is not None and (self.l_sc_fape != 0.0 or self.l_chi != 0.0):
            if "teacher_atom14" not in batch or "teacher_chi" not in batch:
                raise RuntimeError(
                    "side-chain loss is enabled but the batch has no sidechain targets "
                    "(teacher_atom14/teacher_chi). Re-run preprocessing with --sidechains. "
                    "Refusing to train a silently-disabled loss.")
            if self.l_sc_fape != 0.0:
                gt = sidechain_gt_from_atom14(batch["aatype"], batch["teacher_atom14"],
                                              batch["teacher_atom14_mask"])
                # padded residues give degenerate (N,CA,C) -> non-finite frames; they are
                # masked out by rigidgroups_gt_exists, but 0*NaN=NaN, so sanitise first.
                gt["rigidgroups_gt_frames"] = torch.nan_to_num(gt["rigidgroups_gt_frames"])
                gt["rigidgroups_alt_gt_frames"] = torch.nan_to_num(
                    gt["rigidgroups_alt_gt_frames"])
                gt["rigidgroups_gt_exists"] = gt["rigidgroups_gt_exists"] * \
                    seq_mask[..., None]
                ren = compute_renamed_ground_truth(gt, aux["atom14"])
                sc_fape = sidechain_loss(
                    aux["sidechain_frames"][None], aux["atom14"][None],
                    rigidgroups_gt_frames=gt["rigidgroups_gt_frames"],
                    rigidgroups_alt_gt_frames=gt["rigidgroups_alt_gt_frames"],
                    rigidgroups_gt_exists=gt["rigidgroups_gt_exists"],
                    renamed_atom14_gt_positions=ren["renamed_atom14_gt_positions"],
                    renamed_atom14_gt_exists=ren["renamed_atom14_gt_exists"],
                    alt_naming_is_better=ren["alt_naming_is_better"],
                    clamp_distance=self.sc_clamp_distance,
                    length_scale=self.sc_length_scale).mean()
            if self.l_chi != 0.0:
                chi_loss = supervised_chi_loss(
                    aux["angles"][None], aux["unnormalized_angles"][None],
                    aatype=batch["aatype"], seq_mask=seq_mask,
                    chi_mask=batch["teacher_chi_mask"] * seq_mask[..., None],
                    chi_angles_sin_cos=batch["teacher_chi"],
                    chi_weight=self.chi_weight,
                    angle_norm_weight=self.angle_norm_weight).mean()
            total = total + self.l_sc_fape * sc_fape + self.l_chi * chi_loss

        # peptide-ONLY loss values, every step (observation; not in `total`).
        with torch.no_grad():
            pep_fape = _fape_be(pep[:, None, :]).mean()        # FAPE at peptide pos
            pep_plddt_ce = _masked_ce(plddt_logits, plddt_bins, pep)
            pep_pae_ce = (_masked_ce(pae_logits, pae_bins, pep_pair * pair_mask)
                          if (self.l_pae != 0.0 and pae_logits is not None)
                          else fape.new_zeros(()))

        terms = {"total": total.detach(), "fape": fape.detach(),
                 "plddt_ce": ce_plddt.detach(), "pae_ce": ce_pae.detach(),
                 "sc_fape": sc_fape.detach(), "chi_loss": chi_loss.detach(),
                 "pep_fape": pep_fape.detach(), "pep_plddt_ce": pep_plddt_ce.detach(),
                 "pep_pae_ce": pep_pae_ce.detach()}
        if self.groove_aware:
            terms["contain"] = contain.detach()
            terms["buried_frac"] = buried_mean.detach()
        return total, terms


# ---- peptide / segment helpers (shared by loss + metrics) ----------------- #
def peptide_mask_from_batch(seq_mask: torch.Tensor,
                            segment_id: torch.Tensor) -> torch.Tensor:
    """[B,N] float mask of peptide residues. The peptide is always the LAST
    segment (id 1 for class I, 2 for class II), so it is the highest valid
    segment id per example — robust to class and to ordering."""
    m = seq_mask.bool()
    seg_valid = segment_id.masked_fill(~m, -1)
    pep_seg = seg_valid.max(dim=1, keepdim=True).values          # [B,1]
    return ((segment_id == pep_seg) & m).float()


# ---- metrics (eval-only) -------------------------------------------------- #
def _kabsch_rot(pred0: torch.Tensor, target0: torch.Tensor) -> torch.Tensor:
    """Optimal rotation mapping centered pred0 -> centered target0 (Kabsch)."""
    u, _, vt = torch.linalg.svd(pred0.transpose(0, 1) @ target0)
    d = torch.sign(torch.linalg.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
    diag = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    return vt.transpose(0, 1) @ diag @ u.transpose(0, 1)


def _superpose_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Kabsch-aligned Cα RMSD for one example's masked points [M,3]."""
    mu_p, mu_t = pred.mean(0), target.mean(0)
    p0, t0 = pred - mu_p, target - mu_t
    aligned = p0 @ _kabsch_rot(p0, t0).transpose(0, 1)
    return torch.sqrt(((aligned - t0) ** 2).sum(-1).mean())


def _superpose_rmsd_on(pred: torch.Tensor, target: torch.Tensor,
                       align: torch.Tensor, evalm: torch.Tensor
                       ) -> Optional[torch.Tensor]:
    """RMSD over the ``evalm`` points after superposing on the ``align`` points —
    the pMHC binding-pose metric (align on MHC groove, measure the peptide).
    ``align``/``evalm`` are boolean masks over the points. None if too few points."""
    if int(align.sum()) < 3 or int(evalm.sum()) < 1:
        return None
    pa, ta = pred[align], target[align]
    mu_p, mu_t = pa.mean(0), ta.mean(0)
    rot = _kabsch_rot(pa - mu_p, ta - mu_t)
    pe = (pred[evalm] - mu_p) @ rot.transpose(0, 1)
    te = target[evalm] - mu_t
    return torch.sqrt(((pe - te) ** 2).sum(-1).mean())


def kabsch_rmsd(pred_ca: torch.Tensor, target_ca: torch.Tensor,
                seq_mask: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of per-example superposed Cα RMSD (masked)."""
    vals = []
    for b in range(pred_ca.shape[0]):
        m = seq_mask[b].bool()
        vals.append(_superpose_rmsd(pred_ca[b][m], target_ca[b][m]))
    return torch.stack(vals).mean()


def plddt_from_logits(plddt_logits: torch.Tensor, no_bins: int) -> torch.Tensor:
    centers = (torch.arange(no_bins, device=plddt_logits.device) + 0.5) / no_bins
    return (plddt_logits.softmax(-1) * centers).sum(-1) * 100.0


def pae_from_logits(pae_logits: torch.Tensor, breaks: torch.Tensor) -> torch.Tensor:
    step = breaks[1] - breaks[0]
    centers = torch.cat([breaks + step / 2, (breaks[-1] + step)[None]])
    return (pae_logits.softmax(-1) * centers).sum(-1)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar, br = ar - ar.mean(), br - br.mean()
    return (ar * br).sum() / (ar.norm() * br.norm() + 1e-8)


@torch.no_grad()
def eval_metrics(ca: torch.Tensor, plddt_logits: torch.Tensor,
                 pae_logits: torch.Tensor, batch: Dict[str, torch.Tensor],
                 loss_mod: DistillLoss) -> Dict[str, float]:
    seq_mask = batch["seq_mask"]
    pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
    m = seq_mask.bool()
    rmsd = kabsch_rmsd(ca, batch["teacher_ca"], seq_mask)
    plddt_pred = plddt_from_logits(plddt_logits, loss_mod.plddt_no_bins)
    spearman = _spearman(plddt_pred[m], batch["teacher_plddt"][m])
    if pae_logits is None:                       # model has no PAE head
        pae_mae = ca.new_zeros(())
    else:
        pae_pred = pae_from_logits(pae_logits, loss_mod.pae_breaks)
        pae_mae = ((pae_pred - batch["teacher_pae"]).abs() * pair_mask).sum() \
            / pair_mask.sum().clamp_min(1.0)

    # ---- peptide-only observation metrics (the part that actually matters) ---
    pep = peptide_mask_from_batch(seq_mask, batch["segment_id"])      # [B,N]
    mhc = seq_mask * (1.0 - pep)
    pep_b = pep.bool()
    # peptide Cα-RMSD superposed on the MHC (binding pose), averaged over examples
    pep_rmsds = []
    for b in range(ca.shape[0]):
        r = _superpose_rmsd_on(ca[b], batch["teacher_ca"][b],
                               mhc[b].bool(), pep_b[b])
        if r is not None:
            pep_rmsds.append(r)
    pep_ca_rmsd = float(torch.stack(pep_rmsds).mean()) if pep_rmsds else float("nan")
    # peptide pLDDT MAE and peptide-involving PAE MAE
    pep_plddt_mae = float(((plddt_pred - batch["teacher_plddt"]).abs() * pep).sum()
                          / pep.sum().clamp_min(1.0))
    pep_pair = torch.maximum(pep[:, :, None], pep[:, None, :]) * pair_mask
    pep_pae_mae = 0.0 if pae_logits is None else float(
        ((pae_pred - batch["teacher_pae"]).abs() * pep_pair).sum()
        / pep_pair.sum().clamp_min(1.0))
    return {"ca_rmsd": float(rmsd), "plddt_spearman": float(spearman),
            "pae_mae": float(pae_mae), "pep_ca_rmsd": pep_ca_rmsd,
            "pep_plddt_mae": pep_plddt_mae, "pep_pae_mae": pep_pae_mae}


# =========================================================================== #
# PART 2 — train / eval helpers (heavy lifting; train.py is a thin CLI)
# =========================================================================== #
from collections import defaultdict           # noqa: E402
from torch.utils.data import DataLoader        # noqa: E402


def make_dataloader(dataset: DistillDataset, batch_size: int, shuffle: bool,
                    num_workers: int = 0, sampler=None) -> DataLoader:
    """DataLoader using ``collate_with_teacher``. ``sampler`` lets a caller plug
    in a DistributedSampler for DDP (single-GPU default leaves it None)."""
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_with_teacher,
                      drop_last=False)


def move_batch(batch: Dict[str, object], device: str) -> Dict[str, object]:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# --- metric logging (TensorBoard if available, always a flat CSV) ---------- #
_METRIC_COLS = ["wall_time", "split", "epoch", "global_step", "lr",
                "total", "fape", "plddt_ce", "pae_ce", "sc_fape", "chi_loss",
                "pep_fape", "pep_plddt_ce", "pep_pae_ce",
                "ca_rmsd", "plddt_spearman", "pae_mae",
                "pep_ca_rmsd", "pep_plddt_mae", "pep_pae_mae",
                "contain", "buried_frac", "it_per_s"]


class MetricLogger:
    """Writes scalars for later plotting. Always appends to ``<run_dir>/metrics.csv``
    (dependency-free, load with ``pandas.read_csv``); additionally streams to
    TensorBoard at ``<run_dir>/tb`` if the ``tensorboard`` package is importable
    (``tensorboard --logdir <ckpt_dir>``)."""

    def __init__(self, run_dir: Optional[Path], enable_tb: bool = True):
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.csv_path = (self.run_dir / "metrics.csv") if self.run_dir else None
        self.writer = None
        # Use an existing file's header so a resumed run stays column-aligned;
        # otherwise start a fresh file with the current schema.
        if self.csv_path is not None and self.csv_path.exists():
            self._cols = self.csv_path.read_text().splitlines()[0].split(",")
        else:
            self._cols = list(_METRIC_COLS)
            if self.csv_path is not None:
                self.csv_path.write_text(",".join(self._cols) + "\n")
        if enable_tb and self.run_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.run_dir / "tb"))
            except Exception:                          # tensorboard not installed
                self.writer = None

    def log(self, split: str, epoch: int, step: int, lr: Optional[float],
            metrics: Dict[str, float], it_per_s: Optional[float] = None) -> None:
        import time
        row = {"wall_time": time.time(), "split": split, "epoch": epoch,
               "global_step": step, "lr": lr, "it_per_s": it_per_s, **metrics}
        if self.csv_path is not None:
            line = ",".join("" if row.get(c) is None else f"{row.get(c)}"
                            for c in self._cols)
            with open(self.csv_path, "a") as f:
                f.write(line + "\n")
        if self.writer is not None:
            for k, v in metrics.items():
                if v is not None:
                    self.writer.add_scalar(f"{split}/{k}", float(v), step)
            if lr is not None:
                self.writer.add_scalar("train/lr", float(lr), step)
            if it_per_s is not None:
                self.writer.add_scalar("train/it_per_s", float(it_per_s), step)
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def make_epoch_loader(dataset, batch_size: int, num_workers: int, seed: int,
                      epoch: int, skip_batches: int = 0,
                      rank: int = 0, world_size: int = 1) -> DataLoader:
    """Deterministic per-epoch shuffle (so a mid-epoch resume can replay the exact
    order). ``skip_batches`` drops the first N batches *without loading them* by
    slicing the precomputed permutation — used to fast-forward on resume.

    For DDP (``world_size>1``) the permutation is truncated to a multiple of
    ``world_size`` and strided by ``rank``, so every rank gets an EQUAL-size disjoint
    shard (equal step counts -> no collective deadlock on the last batch)."""
    g = torch.Generator().manual_seed((seed + 1) * 1_000_003 + epoch)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    if world_size > 1:
        usable = (len(perm) // world_size) * world_size
        perm = perm[:usable][rank::world_size]
    if skip_batches > 0:
        perm = perm[skip_batches * batch_size:]
    return make_dataloader(dataset, batch_size, shuffle=False,
                           num_workers=num_workers, sampler=perm)


def train_one_epoch(model: DistillModel, loader: DataLoader,
                    loss_mod: DistillLoss, optimizer, scheduler, device: str,
                    scaler=None, grad_clip: Optional[float] = None,
                    log=None, log_every: int = 0, epoch: int = 0,
                    global_step: int = 0, mlog: Optional[MetricLogger] = None,
                    ckpt_every: int = 0, save_fn=None,
                    recycle_probs: Optional[Sequence[float]] = None,
                    core: Optional[DistillModel] = None, is_main: bool = True,
                    amp_dtype=None, metrics_every: int = 0
                    ) -> Tuple[Dict[str, float], int]:
    """One epoch of encoder-only training. Returns (epoch-mean of each loss term,
    updated global_step).

    Every ``log_every`` steps it prints ALL loss terms (window-averaged since the
    last log) + lr + it/s, and records them to ``mlog`` (TensorBoard + CSV). Every
    ``ckpt_every`` steps it calls ``save_fn(global_step, epoch)`` to checkpoint, so
    a crash mid-epoch loses at most ``ckpt_every`` steps.

    ``metrics_every`` > 0 additionally computes the geometric metrics (peptide Cα-RMSD,
    pLDDT Spearman, …) on the TRAINING batch every that-many steps and folds their
    window-mean into the logged line/CSV — so you can watch pep-RMSD/pLDDT-corr on
    train, not only the loss. These are on the train-mode forward (dropout + MHC noise
    on), so read them as a trend, not the clean-input number the val pass reports."""
    import time
    core = core or model                 # unwrapped module (DDP -> .module) for
    model.train()                        # attribute/param access; `model` (maybe DDP)
    use_amp = scaler is not None and scaler.is_enabled()   # is what we call forward on
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    agg: Dict[str, float] = defaultdict(float)        # whole-epoch (returned)
    wagg: Dict[str, float] = defaultdict(float)       # window since last log
    mwagg: Dict[str, float] = defaultdict(float)      # geometric-metric window
    n = 0
    wn = 0
    mwn = 0
    nsteps = len(loader)
    t0 = time.perf_counter()
    tw = t0
    for i, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        # sample the recycle count per step. With recycle_probs=[p0,p1,...] draw
        # nr from that categorical (e.g. [0.8,0.2] -> 80% no recycle, 20% one);
        # otherwise AF2-style uniform over 1..recycles (0 if the model has none).
        max_rc = getattr(core, "recycles", 0)
        if recycle_probs:
            nr = random.choices(range(len(recycle_probs)), weights=recycle_probs)[0]
        else:
            nr = random.randint(1, max_rc) if max_rc >= 1 else 0
        # amp_dtype=bfloat16 -> autocast in bf16 WITHOUT a GradScaler (bf16 has fp32's
        # exponent range, so no overflow and no loss scaling needed). This is the path
        # AF2/OpenFold needs — fp16 overflows the Evoformer into NaN. fp16 (the
        # original behaviour) still runs when a scaler is enabled.
        with torch.autocast(device_type=dev_type,
                            enabled=(use_amp or amp_dtype is not None),
                            dtype=(amp_dtype or torch.float16)):
            o = model(batch, return_frames=True, num_recycles=nr)
            ca, plddt, pae, frames = o[:4]      # models with a torsion head return a
            aux = o[4] if len(o) > 4 else None  # 5th `aux` (angles/atom14/sc frames)
            total, terms = loss_mod(ca, plddt, pae, frames, batch, aux=aux)
        if use_amp:
            scaler.scale(total).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(core.trainable_parameters(),
                                               grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP may skip the optimizer step while calibrating the loss scale
            # (inf grads -> scale reduced); only advance the LR schedule when the
            # step actually happened, else PyTorch warns and skips an LR value.
            stepped = scaler.get_scale() >= prev_scale
        else:
            total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(core.trainable_parameters(),
                                               grad_clip)
            optimizer.step()
            stepped = True
        if stepped:
            scheduler.step()
        global_step += 1
        bs = int(batch["aatype"].shape[0])
        for k, v in terms.items():
            fv = float(v) * bs
            agg[k] += fv
            wagg[k] += fv
        n += bs
        wn += bs
        # geometric metrics on the training batch (pep RMSD, pLDDT Spearman, …)
        if metrics_every and (i % metrics_every == 0 or i == nsteps):
            with torch.no_grad():
                mets = eval_metrics(ca.detach().float(), plddt.detach().float(),
                                    pae.detach().float(), batch, loss_mod)
            for k, v in mets.items():
                mwagg[k] += float(v) * bs
            mwn += bs
        if log_every and (i % log_every == 0 or i == nsteps):
            now = time.perf_counter()
            rate = log_every / max(1e-9, now - tw) if i % log_every == 0 \
                else (i % log_every) / max(1e-9, now - tw)
            lr = optimizer.param_groups[0]["lr"]
            means = {k: wagg[k] / max(wn, 1) for k in wagg}
            if mwn > 0:                                    # fold in geometric metrics
                for k in mwagg:
                    means[k] = mwagg[k] / mwn
                mwagg.clear()
                mwn = 0
            if log:
                log(f"[train]   epoch {epoch} step {i}/{nsteps} "
                    f"(gstep {global_step}) | "
                    f"total {means.get('total', 0):.3f} "
                    f"fape {means.get('fape', 0):.3f} "
                    f"plddt {means.get('plddt_ce', 0):.3f} "
                    + (f"sc_fape {means['sc_fape']:.3f} chi {means['chi_loss']:.3f}"
                       if (means.get('sc_fape') or means.get('chi_loss'))
                       else f"pae {means.get('pae_ce', 0):.3f}")
                    + f" | pep: "
                    f"fape {means.get('pep_fape', 0):.3f} "
                    f"plddt {means.get('pep_plddt_ce', 0):.3f} "
                    f"pae {means.get('pep_pae_ce', 0):.3f}"
                    + (f" | pep-RMSD {means['pep_ca_rmsd']:.3f} "
                       f"pLDDT-corr {means.get('plddt_spearman', 0):.3f}"
                       if 'pep_ca_rmsd' in means else "")
                    + (f" | contain {means.get('contain', 0):.3f} "
                       f"buried {means.get('buried_frac', 0):.2f}"
                       if 'contain' in means else "")
                    + f" | lr {lr:.2e} | {rate:.1f} it/s")
            if mlog is not None:
                mlog.log("train", epoch, global_step, lr, means, it_per_s=rate)
            wagg.clear()
            wn = 0
            tw = now
        if is_main and ckpt_every and save_fn is not None \
                and global_step % ckpt_every == 0:
            save_fn(global_step, epoch)
    return {k: v / max(n, 1) for k, v in agg.items()}, global_step


@torch.no_grad()
def evaluate(model: DistillModel, loader: DataLoader, loss_mod: DistillLoss,
             device: str, max_batches: Optional[int] = None,
             num_recycles: Optional[int] = None, return_n: bool = False):
    """Example-weighted mean loss terms + metrics over (part of) a loader.
    ``num_recycles`` overrides the model's default recycle count at eval.
    With ``return_n=True`` also returns the #examples seen, so callers can do an
    example-weighted all-reduce across ranks (distributed validation)."""
    model.eval()
    agg: Dict[str, float] = defaultdict(float)
    n = 0
    for i, batch in enumerate(loader):
        batch = move_batch(batch, device)
        o = model(batch, return_frames=True, num_recycles=num_recycles)
        ca, plddt, pae, frames = o[:4]
        aux = o[4] if len(o) > 4 else None
        _, terms = loss_mod(ca, plddt, pae, frames, batch, aux=aux)
        mets = eval_metrics(ca, plddt, pae, batch, loss_mod)
        bs = int(batch["aatype"].shape[0])
        for d in (terms, mets):
            for k, v in d.items():
                agg[k] += float(v) * bs
        n += bs
        if max_batches is not None and i + 1 >= max_batches:
            break
    model.train()
    means = {k: v / max(n, 1) for k, v in agg.items()}
    return (means, n) if return_n else means


def setup_distributed():
    """Init torch.distributed from the env torchrun sets (WORLD_SIZE/RANK/LOCAL_RANK).
    Returns (distributed, rank, local_rank, world_size); a no-op single-GPU tuple
    when WORLD_SIZE<=1 so the same code path runs on one GPU."""
    import os
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        # Long timeout: rank 0 runs the full validation between epochs (~15+ min on
        # the frozen AF2 stack) while the other ranks wait at the next collective.
        # NCCL's default 10-min watchdog would abort them mid-eval -> raise it.
        import datetime
        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(minutes=60))
    return True, rank, local_rank, world


def run_training(*, variant: int = 7, scheme: str = "two_axis", fold: int = 1,
                 dummy: bool = False, epochs: int = 10, bs: int = 2,
                 lr: float = 1e-3, lambdas: Sequence[float] = (1.0, 0.1, 0.1),
                 seed: int = 0, device: Optional[str] = None, amp: bool = False,
                 grad_clip: Optional[float] = 1.0, weight_decay: float = 1e-4,
                 num_workers: int = 0, af_root: Optional[Path] = None,
                 h5_dir: Optional[Path] = None, ckpt_dir: Optional[Path] = None,
                 run_name: Optional[str] = None, resume: Optional[Path] = None,
                 eval_batches: Optional[int] = None, log_every: int = 2000,
                 ckpt_every: int = 2000, tensorboard: bool = True,
                 peptide_weight: float = 1.0, recycles: int = 0,
                 recycle_probs: Optional[Sequence[float]] = None,
                 eval_recycles: Optional[int] = None,
                 unfreeze_sm: float = 0.0, unfreeze_plddt: float = 0.0,
                 unfreeze_pae: float = 0.0, anchor_relpos: bool = False,
                 plddt_weight_struct: bool = False,
                 plddt_weight_floor: float = 0.1, groove_aware: bool = False,
                 lambda_contain: float = 0.5,
                 groove=(8.0, 1.5, 8.0, 18.0), log=print):
    """Full single-GPU training loop (DDP-ready: pass a sampler via make_dataloader
    and wrap the encoder in DDP externally). Returns (history, model).

    Data source: ``h5_dir`` (the preprocessed streamable store; fastest) else the
    on-the-fly PDB readers (``dummy`` -> data/test/, else ``af_root``).

    Checkpointing: if ``ckpt_dir`` is set, writes ``<ckpt_dir>/<run_name>/last.pt``
    every epoch (encoder + optimizer + scheduler + scaler + epoch + history, so it
    is resumable) and ``best.pt`` (encoder, lowest val Cα-RMSD). ``resume`` (a
    last.pt) continues a preempted run. ``run_name`` defaults to
    ``variant{v}_{scheme}_fold{fold}`` so a SLURM array over variants is isolated.
    """
    set_seed(seed)
    # Multi-GPU via DistributedDataParallel (torchrun sets the env). One process per
    # GPU; each rank trains on a disjoint data shard, grads are all-reduced, only
    # rank 0 logs/evaluates/checkpoints. Single-GPU (WORLD_SIZE<=1) is a no-op.
    distributed, rank, local_rank, world = setup_distributed()
    is_main = (rank == 0)
    if distributed:
        device = f"cuda:{local_rank}"
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not is_main:                       # silence non-main ranks
        log = (lambda *a, **k: None)

    if h5_dir is not None:
        train_ds = build_h5_dataset(h5_dir, scheme, fold, "train", all_ids=dummy)
        val_ds = build_h5_dataset(h5_dir, scheme, fold, "val", all_ids=dummy)
    else:
        train_ds = build_dataset(scheme, fold, "train", dummy=dummy, af_root=af_root)
        val_ds = build_dataset(scheme, fold, "val", dummy=dummy, af_root=af_root)
    # train loader is rebuilt per epoch (deterministic shuffle) so a mid-epoch
    # resume can fast-forward to the exact batch; val loader is fixed.
    val_loader = make_dataloader(val_ds, bs, shuffle=False,
                                 num_workers=num_workers)
    # per-RANK steps (data is sharded across ranks under DDP) -> the loop, scheduler
    # horizon and resume math all count per-rank steps.
    shard_n = len(train_ds) // world if world > 1 else len(train_ds)
    steps_per_epoch = (shard_n + bs - 1) // bs

    model = DistillModel(variant, device=device, recycles=recycles,
                         unfreeze_sm=unfreeze_sm, unfreeze_plddt=unfreeze_plddt,
                         unfreeze_pae=unfreeze_pae, anchor_relpos=anchor_relpos).to(device)
    loss_mod = DistillLoss(*lambdas, peptide_weight=peptide_weight,
                           plddt_weight_struct=plddt_weight_struct,
                           plddt_weight_floor=plddt_weight_floor,
                           groove_aware=groove_aware,
                           lambda_contain=lambda_contain, groove=groove).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr,
                                  weight_decay=weight_decay)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp)

    import datetime
    config = {
        "variant": variant, "scheme": scheme, "fold": fold, "dummy": dummy,
        "lambdas": tuple(lambdas), "peptide_weight": peptide_weight,
        "recycles": recycles, "anchor_relpos": anchor_relpos,
        "plddt_weight_struct": plddt_weight_struct,
        "plddt_weight_floor": plddt_weight_floor,
        "groove_aware": groove_aware, "lambda_contain": lambda_contain,
        "groove": tuple(groove),
        "unfreeze_sm": unfreeze_sm,
        "unfreeze_plddt": unfreeze_plddt, "unfreeze_pae": unfreeze_pae,
        "unfrozen_fraction": model.unfrozen,
        "bs": bs, "lr": lr, "epochs": epochs,
        "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
        "seed": seed, "num_workers": num_workers, "device": str(device),
        "h5_dir": str(h5_dir) if h5_dir else None,
        "af_root": str(af_root) if af_root else None,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "trainable_params": sum(p.numel() for p in model.trainable_parameters()),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    run_dir = None
    if ckpt_dir is not None:
        run_name = run_name or f"variant{variant}_{scheme}_fold{fold}"
        config["run_name"] = run_name
        run_dir = Path(ckpt_dir) / run_name
        if is_main:                       # only rank 0 writes config / checkpoints
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # metric logging / checkpoint writing on rank 0 only (None -> no-op elsewhere)
    mlog = MetricLogger(run_dir if is_main else None, enable_tb=tensorboard)

    def save_ckpt(path: Path, global_step: int, epoch: int) -> None:
        """Atomic checkpoint (tmp + rename) holding everything needed to resume."""
        if run_dir is None or not is_main:           # rank 0 only -> no write races
            return
        tmp = Path(str(path) + ".tmp")
        torch.save({"epoch": epoch, "global_step": global_step,
                    "encoder": model.encoder.state_dict(),
                    "frozen_trainable": model.frozen_trainable_state(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "history": history,
                    "best": best, "config": config}, tmp)
        tmp.replace(path)

    history: List[dict] = []
    start_epoch, best, global_step, resume_skip = 1, float("inf"), 0, 0
    if resume is not None and Path(resume).exists():
        try:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            model.encoder.load_state_dict(ckpt["encoder"])
            if ckpt.get("frozen_trainable"):       # restore unfrozen SM/head params
                model.load_state_dict(ckpt["frozen_trainable"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            # If --epochs changed since the checkpoint, the loaded scheduler still
            # carries the OLD T_max -> extending epochs would just train at lr~0
            # past the old horizon. Re-size the cosine to the new total_steps
            # (keeping the current step) so the schedule actually spans the run.
            if getattr(scheduler, "T_max", total_steps) != total_steps:
                old_tmax = scheduler.T_max
                scheduler.T_max = total_steps
                log(f"[train] re-sized cosine T_max {old_tmax} -> {total_steps} "
                    f"(epochs changed; lr resumes on the new horizon)")
            scaler.load_state_dict(ckpt["scaler"])
            history = ckpt.get("history", [])
            best = ckpt.get("best", float("inf"))
            if distributed:
                # the per-rank step basis differs from a single-GPU checkpoint, so
                # the saved global_step is not comparable. Resume at the EPOCH
                # boundary and realign the step counter + LR schedule to the new
                # (per-rank) basis.
                start_epoch = int(ckpt["epoch"]) + 1
                resume_skip = 0
                global_step = int(ckpt["epoch"]) * steps_per_epoch
                scheduler.last_epoch = global_step
            else:
                # global_step pinpoints where we stopped; derive epoch + batch to skip.
                # (Older epoch-only checkpoints fall back to an epoch boundary.)
                global_step = int(ckpt.get("global_step",
                                           int(ckpt["epoch"]) * steps_per_epoch))
                start_epoch = global_step // steps_per_epoch + 1
                resume_skip = global_step % steps_per_epoch
            log(f"[train] resumed from {resume} at epoch {start_epoch} "
                f"(global_step {global_step}, skip {resume_skip} batches, "
                f"best val Cα-RMSD {best:.3f})")
        except Exception as e:
            # corrupt/partial checkpoint (e.g. a hard kill before atomic rename
            # on an older run): don't get stuck forever — start fresh instead.
            log(f"[train] WARNING: could not load {resume} ({type(e).__name__}: "
                f"{e}); starting from scratch")
            history, start_epoch, best, global_step, resume_skip = \
                [], 1, float("inf"), 0, 0

    log(f"[train] variant={variant} scheme={scheme} fold={fold} dummy={dummy} "
        f"device={device} | train={len(train_ds)} val={len(val_ds)} "
        f"bs={bs} epochs={epochs} steps/epoch={steps_per_epoch} "
        f"total_steps={total_steps} amp={amp} ckpt_every={ckpt_every}"
        + (f" | ckpt={run_dir}" if run_dir else ""))
    # wrap for DDP just before the loop (model stays the unwrapped 'core' used for
    # eval/checkpoint/attribute access; 'net' is what we run forward+backward on).
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    else:
        net = model

    for epoch in range(start_epoch, epochs + 1):
        skip = resume_skip if epoch == start_epoch else 0
        train_loader = make_epoch_loader(train_ds, bs, num_workers, seed, epoch,
                                         skip_batches=skip, rank=rank,
                                         world_size=world)
        tr, global_step = train_one_epoch(
            net, train_loader, loss_mod, optimizer, scheduler, device,
            scaler if amp else None, grad_clip, log=(log if is_main else None),
            log_every=log_every, epoch=epoch, global_step=global_step, mlog=mlog,
            ckpt_every=ckpt_every, recycle_probs=recycle_probs,
            core=model, is_main=is_main,
            save_fn=(lambda gs, ep: save_ckpt(run_dir / "last.pt", gs, ep))
            if (run_dir is not None and is_main) else None)
        # evaluation + checkpoint on rank 0 only; other ranks loop to the next epoch
        # and wait at the first all-reduce.
        if is_main:
            ev = evaluate(model, val_loader, loss_mod, device,
                          max_batches=eval_batches, num_recycles=eval_recycles)
            history.append({"epoch": epoch, "train": tr, "val": ev})
            mlog.log("val", epoch, global_step, None, ev)
            log(f"[train] epoch {epoch:3d} | train total {tr['total']:.3f} "
                f"(fape {tr['fape']:.3f} plddt {tr['plddt_ce']:.3f} "
                f"pae {tr['pae_ce']:.3f} | pep fape {tr.get('pep_fape', 0):.3f} "
                f"plddt {tr.get('pep_plddt_ce', 0):.3f} pae {tr.get('pep_pae_ce', 0):.3f}) "
                f"| val total {ev['total']:.3f} Cα-RMSD {ev['ca_rmsd']:.2f} "
                f"pep-RMSD {ev['pep_ca_rmsd']:.2f} pLDDT-sp {ev['plddt_spearman']:.2f} "
                f"PAE-MAE {ev['pae_mae']:.2f}")
            if run_dir is not None:
                if ev["ca_rmsd"] < best:
                    best = ev["ca_rmsd"]
                    torch.save({"epoch": epoch, "global_step": global_step,
                                "encoder": model.encoder.state_dict(),
                                "frozen_trainable": model.frozen_trainable_state(),
                                "val": ev, "config": config}, run_dir / "best.pt")
                    log(f"[train]   new best val Cα-RMSD {best:.3f} -> best.pt")
                save_ckpt(run_dir / "last.pt", global_step, epoch)
    mlog.close()
    if distributed:
        torch.distributed.destroy_process_group()
    return history, model


# =========================================================================== #
# PART 2 — one-time preprocessing to a streamable HDF5 store
# =========================================================================== #
# Layout: one shard per PMGen chunk, <out>/<chunk>.h5, one group per anchor id,
# plus <out>/<chunk>.index.parquet (merged into <out>/index.parquet). Splits are
# id lists (read_split_ids base ids -> anchor ids), so data is stored once.
#
# Per-id datasets (compact dtypes):
#   aatype u1, residue_index i2, anchor u1, segment_id u1,
#   teacher_ca f4 [N,3], teacher_bb f4 [N,3,3],
#   teacher_plddt f2 [N], teacher_pae f2 [N,N]   (PAE dominates size -> gzip)
# Group attrs: n_mhc, n_pep, mhc_type, base_id, mhc_seq, peptide, anchors,
#   hla_cluster_id, peptide_cluster_id.
_H5_DSETS = ("aatype", "residue_index", "anchor", "segment_id",
             "teacher_ca", "teacher_bb", "teacher_plddt", "teacher_pae")


def extract_example_arrays(pdb: Path, plddt_npy: Path, pae_npy: Path,
                           peptide: str, mhc_seq: str, anchors: str,
                           mhc_type, sidechains: bool = False
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Parse one teacher example into compact numpy arrays + metadata. With
    ``sidechains`` also stores the side-chain torsions ``teacher_chi`` [N,4,2]
    (sin/cos of χ1..χ4) and ``teacher_chi_mask`` [N,4] (which χ exist)."""
    ex = parse_example(pdb, peptide, mhc_seq, anchors, mhc_type,
                       return_backbone=True, return_sidechain=sidechains)
    n = int(ex["aatype"].shape[0])
    plddt, pae = load_teacher_arrays(plddt_npy, pae_npy, n)
    arrays = {
        "aatype": ex["aatype"].numpy().astype(np.uint8),
        "residue_index": ex["residue_index"].numpy().astype(np.int16),
        "anchor": ex["anchor"].numpy().astype(np.uint8),
        "segment_id": ex["segment_id"].numpy().astype(np.uint8),
        "teacher_ca": ex["teacher_ca"].numpy().astype(np.float32),
        "teacher_bb": ex["teacher_bb"].numpy().astype(np.float32),
        "teacher_plddt": plddt.numpy().astype(np.float16),
        "teacher_pae": pae.numpy().astype(np.float16),
    }
    if sidechains:
        arrays["teacher_chi"] = ex["teacher_chi"].numpy().astype(np.float16)
        arrays["teacher_chi_mask"] = ex["teacher_chi_mask"].numpy().astype(np.uint8)
    meta = {"n_mhc": int(ex["n_mhc"]), "n_pep": int(ex["n_pep"]),
            "mhc_type": int(mhc_type), "mhc_seq": mhc_seq, "peptide": peptide,
            "anchors": str(anchors)}
    return arrays, meta


def _chunk_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def preprocess_chunk(chunk_dir: Path, shard_path: Path, index_path: Path,
                     alphafold_subdir: str = "alphafold", output_link: str = "output",
                     compression: str = "gzip", clevel: int = 4,
                     sidechains: bool = False, log=print) -> Dict[str, int]:
    """Process one chunk dir (with chunk.tsv + output/<alphafold_subdir>/<id>/)
    into one HDF5 shard + a per-chunk index parquet. Missing/failed ids are
    skipped (counted). Returns counts."""
    import sys
    import pandas as pd
    from tqdm.auto import tqdm
    rows = pd.read_csv(chunk_dir / "chunk.tsv", sep="\t").to_dict("records")
    af_base = chunk_dir / output_link / alphafold_subdir
    if not af_base.exists():
        af_base = chunk_dir / alphafold_subdir
    recs: List[dict] = []
    n_ok = n_missing = n_fail = 0
    comp = dict(compression=compression, compression_opts=clevel, shuffle=True) \
        if compression else {}
    with h5py.File(shard_path, "w") as h5:
        with tqdm(
            rows,
            total=len(rows),
            desc=chunk_dir.name,
            unit="example",
            disable=not sys.stderr.isatty(),
            leave=True
        ) as pbar:
            for r in pbar:
                aid = str(r["id"])
                af_dir = af_base / aid
                if not af_dir.is_dir():
                    n_missing += 1
                    pbar.set_postfix(ok=n_ok, missing=n_missing, failed=n_fail)
                    continue
                try:
                    pdb, plddt_npy, pae_npy = find_teacher_files(af_dir)
                    arrays, meta = extract_example_arrays(
                        pdb, plddt_npy, pae_npy, r["peptide"], r["mhc_seq"],
                        r["anchors"], r["mhc_type"], sidechains=sidechains)
                except (FileNotFoundError, ValueError) as exc:
                    n_fail += 1
                    pbar.set_postfix(ok=n_ok, missing=n_missing, failed=n_fail)
                    tqdm.write(f"    [skip] {aid}: {str(exc)[:80]}")
                    continue
                g = h5.create_group(aid)
                for k in arrays:                 # _H5_DSETS + optional sidechains
                    g.create_dataset(k, data=arrays[k], **comp)
                for k, v in meta.items():
                    g.attrs[k] = v
                base = base_id(aid)
                g.attrs["base_id"] = base
                for opt in ("hla_cluster_id", "peptide_cluster_id"):
                    g.attrs[opt] = str(r.get(opt, ""))
                recs.append({
                    "id": aid,
                    "base_id": base,
                    "shard": shard_path.name,
                    "n_mhc": meta["n_mhc"],
                    "n_pep": meta["n_pep"],
                    "mhc_type": meta["mhc_type"],
                    "hla_cluster_id": str(r.get("hla_cluster_id", "")),
                    "peptide_cluster_id": str(r.get("peptide_cluster_id", ""))
                })
                n_ok += 1
                pbar.set_postfix(ok=n_ok, missing=n_missing, failed=n_fail)
    # write with an explicit header even when recs is empty (a chunk with no
    # successful example) so merge_indices never hits a header-less file
    index_cols = ["id", "base_id", "shard", "n_mhc", "n_pep", "mhc_type",
                  "hla_cluster_id", "peptide_cluster_id"]
    pd.DataFrame(recs, columns=index_cols).to_csv(index_path, index=False)
    log(f"  {chunk_dir.name}: {n_ok} ok, {n_missing} missing, {n_fail} failed "
        f"-> {shard_path.name}")
    return {"ok": n_ok, "missing": n_missing, "failed": n_fail}


def preprocess_chunks(chunks_dir: Path, out_dir: Path,
                      chunks: Optional[Sequence[str]] = None,
                      overwrite: bool = False, merge: bool = True,
                      alphafold_subdir: str = "alphafold",
                      output_link: str = "output",
                      compression: str = "gzip", clevel: int = 4,
                      sidechains: bool = False, log=print) -> None:
    """Preprocess all (or selected) chunk dirs into HDF5 shards + index. Each
    chunk is independent -> safe to run in parallel jobs (pass disjoint
    --chunks); ``merge`` concatenates per-chunk indices into index.parquet.
    ``sidechains`` additionally stores side-chain torsions (for model_2)."""
    chunks_dir, out_dir = Path(chunks_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dirs = [d for d in chunks_dir.iterdir()
                if d.is_dir() and (d / "chunk.tsv").exists()]
    if chunks:
        want = set(chunks)
        all_dirs = [d for d in all_dirs if d.name in want]
    all_dirs.sort(key=lambda d: _chunk_sort_key(d.name))
    log(f"[preprocess] {len(all_dirs)} chunk(s) -> {out_dir}")
    pbar = tqdm(all_dirs, desc="Preprocessing chunks", unit="chunk")
    for cdir in all_dirs:
        shard = out_dir / f"{cdir.name}.h5"
        idx = out_dir / f"{cdir.name}.index.csv"
        if shard.exists() and idx.exists() and not overwrite:
            log(f"  {cdir.name}: shard exists, skipping")
            continue
        preprocess_chunk(cdir, shard, idx, alphafold_subdir, output_link,
                         compression, clevel, sidechains=sidechains, log=log)
    if merge:
        merge_indices(out_dir, log=log)


def merge_indices(out_dir: Path, log=print) -> Optional[Path]:
    """Concatenate the per-chunk ``*.index.csv`` files in ``out_dir`` into a
    single ``index.csv`` (the training store index). Idempotent — the merged
    ``index.csv`` does not match the ``*.index.csv`` glob, so re-running is safe.
    Per-chunk files that are empty (a chunk with no successful example) are
    skipped and reported. Use after a parallel array preprocessing run; needs
    only ``out_dir``."""
    import pandas as pd
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("*.index.csv"))
    frames: List = []
    skipped: List[str] = []
    for p in parts:
        try:
            df = pd.read_csv(p, dtype=str)
        except pd.errors.EmptyDataError:
            skipped.append(p.name)
            continue
        frames.append(df) if len(df) else skipped.append(p.name)
    if not frames:
        log(f"[merge] no non-empty *.index.csv files in {out_dir}")
        return None
    full = pd.concat(frames, ignore_index=True)
    dest = out_dir / "index.csv"
    full.to_csv(dest, index=False)
    log(f"[merge] {len(full):,} examples across {full['shard'].nunique()} shards "
        f"from {len(frames)} index files ({len(skipped)} empty skipped) -> {dest}")
    if skipped:
        log(f"[merge] empty/no-success chunks: {skipped}")
    return dest


# --- HDF5-backed dataset (the streamable training path) -------------------- #
class H5DistillDataset(Dataset):
    """Reads pre-extracted examples from per-chunk HDF5 shards by anchor id.
    File handles are opened lazily (fork-safe for DataLoader workers)."""

    def __init__(self, ids: Sequence[str], id_to_shard: Dict[str, str],
                 h5_dir: Path):
        if not ids:
            raise ValueError("H5DistillDataset received no ids")
        self.ids = list(ids)
        self.id_to_shard = id_to_shard
        self.h5_dir = Path(h5_dir)
        self._handles: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def _h5(self, shard: str) -> h5py.File:
        h = self._handles.get(shard)
        if h is None:
            h = h5py.File(self.h5_dir / shard, "r")
            self._handles[shard] = h
        return h

    def __getitem__(self, i: int) -> Dict[str, object]:
        aid = self.ids[i]
        g = self._h5(self.id_to_shard[aid])[aid]
        n = g["aatype"].shape[0]
        ex = {
            "aatype": torch.from_numpy(g["aatype"][()].astype(np.int64)),
            "residue_index": torch.from_numpy(g["residue_index"][()].astype(np.int64)),
            "seq_mask": torch.ones(n, dtype=torch.float32),
            "anchor": torch.from_numpy(g["anchor"][()].astype(np.float32)),
            "segment_id": torch.from_numpy(g["segment_id"][()].astype(np.int64)),
            "teacher_ca": torch.from_numpy(g["teacher_ca"][()].astype(np.float32)),
            "teacher_bb": torch.from_numpy(g["teacher_bb"][()].astype(np.float32)),
            "teacher_plddt": torch.from_numpy(g["teacher_plddt"][()].astype(np.float32)),
            "teacher_pae": torch.from_numpy(g["teacher_pae"][()].astype(np.float32)),
            "n_mhc": int(g.attrs["n_mhc"]),
            "n_pep": int(g.attrs["n_pep"]),
            "id": aid,
        }
        if "teacher_chi" in g:                   # present only in side-chain stores
            ex["teacher_chi"] = torch.from_numpy(
                g["teacher_chi"][()].astype(np.float32))
            ex["teacher_chi_mask"] = torch.from_numpy(
                g["teacher_chi_mask"][()].astype(np.float32))
        if "teacher_atom14" in g:                # AF2 sidechain-FAPE targets
            ex["teacher_atom14"] = torch.from_numpy(
                g["teacher_atom14"][()].astype(np.float32))
            ex["teacher_atom14_mask"] = torch.from_numpy(
                g["teacher_atom14_mask"][()].astype(np.float32))
        return ex


def load_h5_index(h5_dir: Path):
    import pandas as pd
    return pd.read_csv(Path(h5_dir) / "index.csv",
                       dtype={"id": str, "base_id": str, "shard": str,
                              "hla_cluster_id": str, "peptide_cluster_id": str})


def build_h5_dataset(h5_dir: Path, scheme: str = "two_axis", fold: int = 1,
                     split: str = "train", ids: Optional[Sequence[str]] = None,
                     all_ids: bool = False, pool_csv: Path = POOL_CSV
                     ) -> H5DistillDataset:
    """Build an H5DistillDataset. Pass ``all_ids=True`` to use every example in
    the store; ``ids=[...]`` for an explicit list; otherwise the anchor ids whose
    **base id** lies in the (scheme, fold, split) partition (so all anchor combos
    of a pair stay together)."""
    idx = load_h5_index(h5_dir)
    id_to_shard = dict(zip(idx["id"], idx["shard"]))
    if all_ids:
        chosen = idx["id"].tolist()
    elif ids is not None:
        want = set(ids)
        chosen = [i for i in idx["id"] if i in want]
    else:
        split_base = set(read_split_ids(scheme, fold, pool_csv)[split])
        chosen = idx.loc[idx["base_id"].isin(split_base), "id"].tolist()
    return H5DistillDataset(chosen, id_to_shard, h5_dir)
