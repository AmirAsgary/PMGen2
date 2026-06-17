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
import re
import sys
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
from openfold.utils.loss import backbone_loss                         # noqa: E402


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
class DistillEncoder(nn.Module):
    """Stripped pairformer encoder: featurize -> ``depth`` blocks -> project to
    the frozen stack's exact widths s:[B,N,384], z:[B,N,N,128]."""

    def __init__(self, variant: int, d_s: int = 384, d_z: int = 128,
                 depth: int = 1, *, c_hidden_mul: int = 128, c_hidden_tri: int = 32,
                 no_heads_tri: int = 4, no_heads_s: int = 8, c_hidden_s: int = 32,
                 transition_factor: int = 2, max_offset: int = 32):
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

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor,
                seq_mask: torch.Tensor, anchor: torch.Tensor,
                segment_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]          # [B,N,N]
        s, z = self.featurizer(aatype, residue_index, anchor, segment_id)
        s = s * seq_mask[..., None]
        z = z * pair_mask[..., None]
        for block in self.blocks:
            s, z = block(s, z, seq_mask, pair_mask)
        s_out = self.s_out(self.s_out_norm(s)) * seq_mask[..., None]
        z_out = self.z_out(self.z_out_norm(z)) * pair_mask[..., None]
        return s_out, z_out


class DistillModel(nn.Module):
    """Trainable encoder + FROZEN AF2 stack. Only the encoder has gradients."""

    def __init__(self, variant: int, model_name: str = "model_2_ptm",
                 device: str = "cpu", **encoder_kwargs):
        super().__init__()
        self.encoder = DistillEncoder(variant, **encoder_kwargs)
        self.frozen = load_frozen_fold(model_name, device=device)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.to(device)

    def train(self, mode: bool = True) -> "DistillModel":
        super().train(mode)
        self.frozen.eval()                     # frozen stack never trains
        return self

    def trainable_parameters(self):
        return self.encoder.parameters()

    def forward(self, batch: Dict[str, torch.Tensor], return_frames: bool = False):
        """-> (ca[B,N,3], plddt_logits[B,N,50], pae_logits[B,N,N,64])
        and, if ``return_frames``, the SM backbone-frame trajectory [n_blocks,B,N,7]
        for FAPE. Uses the frozen submodules directly (identical math to
        FrozenFold.forward) so the frames are available without a second pass."""
        s, z = self.encoder(batch["aatype"], batch["residue_index"],
                            batch["seq_mask"], batch["anchor"],
                            batch["segment_id"])
        out = self.frozen.sm({"single": s, "pair": z}, batch["aatype"],
                             mask=batch["seq_mask"])
        ca = out["positions"][-1][..., 1, :]
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
    for i, e in enumerate(examples):
        n = int(e["aatype"].shape[0])
        teacher_plddt[i, :n] = e["teacher_plddt"]
        teacher_pae[i, :n, :n] = e["teacher_pae"]
        if has_bb:
            teacher_bb[i, :n] = e["teacher_bb"]
    batch["teacher_plddt"] = teacher_plddt
    batch["teacher_pae"] = teacher_pae
    if has_bb:
        batch["teacher_bb"] = teacher_bb
    if "id" in examples[0]:
        batch["id"] = [e["id"] for e in examples]
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


class DistillLoss(nn.Module):
    """L = λ_fape·FAPE + λ_plddt·CE(plddt, bin50) + λ_pae·CE(pae, bin64).

    The CE terms backprop through the (frozen, ungradiented) heads + SM into the
    encoder. Bin breakpoints come from the frozen model's loss config, not hard-
    coded: PAE no_bins/max_bin from ``loss.tm``; pLDDT no_bins from ``heads.lddt``.
    """

    def __init__(self, lambda_fape: float = 1.0, lambda_plddt: float = 0.1,
                 lambda_pae: float = 0.1, model_name: str = "model_2_ptm",
                 fape_clamp: Optional[float] = None, fape_unit: float = 10.0):
        # fape_clamp=None -> unclamped FAPE: needed to fold from a random init
        # (a 10 Å clamp zeroes the gradient when every error exceeds it). AF2's
        # clamped/clamp-schedule variant can be enabled once structures are close.
        super().__init__()
        self.l_fape, self.l_plddt, self.l_pae = lambda_fape, lambda_plddt, lambda_pae
        self.fape_clamp, self.fape_unit = fape_clamp, fape_unit
        cfg = model_config(model_name)
        self.pae_no_bins = int(cfg["loss"]["tm"]["no_bins"])
        self.plddt_no_bins = int(cfg["model"]["heads"]["lddt"]["no_bins"])
        pae_max = float(cfg["loss"]["tm"]["max_bin"])
        self.register_buffer(
            "pae_breaks", torch.linspace(0.0, pae_max, self.pae_no_bins - 1))

    def forward(self, ca: torch.Tensor, plddt_logits: torch.Tensor,
                pae_logits: torch.Tensor, frames: torch.Tensor,
                batch: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_mask = batch["seq_mask"]
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]

        bb_tensor, bb_mask = _build_gt_backbone_frames(
            batch["aatype"], batch["teacher_bb"], seq_mask)
        fape = backbone_loss(backbone_rigid_tensor=bb_tensor,
                             backbone_rigid_mask=bb_mask, traj=frames,
                             clamp_distance=self.fape_clamp,
                             loss_unit_distance=self.fape_unit)
        fape = fape.mean() if fape.ndim > 0 else fape

        ce_plddt = _masked_ce(plddt_logits,
                              bin_plddt(batch["teacher_plddt"], self.plddt_no_bins),
                              seq_mask)
        ce_pae = _masked_ce(pae_logits,
                            torch.bucketize(batch["teacher_pae"], self.pae_breaks),
                            pair_mask)
        total = self.l_fape * fape + self.l_plddt * ce_plddt + self.l_pae * ce_pae
        terms = {"total": total.detach(), "fape": fape.detach(),
                 "plddt_ce": ce_plddt.detach(), "pae_ce": ce_pae.detach()}
        return total, terms


# ---- metrics (eval-only) -------------------------------------------------- #
def _superpose_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Kabsch-aligned Cα RMSD for one example's masked points [M,3]."""
    mu_p, mu_t = pred.mean(0), target.mean(0)
    p0, t0 = pred - mu_p, target - mu_t
    u, _, vt = torch.linalg.svd(p0.transpose(0, 1) @ t0)
    d = torch.sign(torch.linalg.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
    diag = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    rot = vt.transpose(0, 1) @ diag @ u.transpose(0, 1)
    aligned = p0 @ rot.transpose(0, 1)
    return torch.sqrt(((aligned - t0) ** 2).sum(-1).mean())


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
    pae_pred = pae_from_logits(pae_logits, loss_mod.pae_breaks)
    pae_mae = ((pae_pred - batch["teacher_pae"]).abs() * pair_mask).sum() \
        / pair_mask.sum().clamp_min(1.0)
    return {"ca_rmsd": float(rmsd), "plddt_spearman": float(spearman),
            "pae_mae": float(pae_mae)}


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


def train_one_epoch(model: DistillModel, loader: DataLoader,
                    loss_mod: DistillLoss, optimizer, scheduler, device: str,
                    scaler=None, grad_clip: Optional[float] = None,
                    log=None, log_every: int = 0, epoch: int = 0
                    ) -> Dict[str, float]:
    """One epoch of encoder-only training; returns example-weighted mean of each
    loss term. With ``log_every > 0`` prints a running summary every that many
    steps (long epochs otherwise emit nothing for hours)."""
    import time
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    agg: Dict[str, float] = defaultdict(float)
    n = 0
    nsteps = len(loader)
    t0 = time.perf_counter()
    for i, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, enabled=use_amp):
            ca, plddt, pae, frames = model(batch, return_frames=True)
            total, terms = loss_mod(ca, plddt, pae, frames, batch)
        if use_amp:
            scaler.scale(total).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
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
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                               grad_clip)
            optimizer.step()
            stepped = True
        if stepped:
            scheduler.step()
        bs = int(batch["aatype"].shape[0])
        for k, v in terms.items():
            agg[k] += float(v) * bs
        n += bs
        if log and log_every and (i % log_every == 0 or i == nsteps):
            rate = i / max(1e-9, time.perf_counter() - t0)
            lr = optimizer.param_groups[0]["lr"]
            log(f"[train]   epoch {epoch} step {i}/{nsteps} | "
                f"total {agg['total'] / max(n, 1):.3f} "
                f"fape {agg['fape'] / max(n, 1):.3f} | "
                f"lr {lr:.2e} | {rate:.1f} it/s")
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate(model: DistillModel, loader: DataLoader, loss_mod: DistillLoss,
             device: str, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Example-weighted mean loss terms + metrics over (part of) a loader."""
    model.eval()
    agg: Dict[str, float] = defaultdict(float)
    n = 0
    for i, batch in enumerate(loader):
        batch = move_batch(batch, device)
        ca, plddt, pae, frames = model(batch, return_frames=True)
        _, terms = loss_mod(ca, plddt, pae, frames, batch)
        mets = eval_metrics(ca, plddt, pae, batch, loss_mod)
        bs = int(batch["aatype"].shape[0])
        for d in (terms, mets):
            for k, v in d.items():
                agg[k] += float(v) * bs
        n += bs
        if max_batches is not None and i + 1 >= max_batches:
            break
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def run_training(*, variant: int = 7, scheme: str = "two_axis", fold: int = 1,
                 dummy: bool = False, epochs: int = 10, bs: int = 2,
                 lr: float = 1e-3, lambdas: Sequence[float] = (1.0, 0.1, 0.1),
                 seed: int = 0, device: Optional[str] = None, amp: bool = False,
                 grad_clip: Optional[float] = 1.0, weight_decay: float = 1e-4,
                 num_workers: int = 0, af_root: Optional[Path] = None,
                 h5_dir: Optional[Path] = None, ckpt_dir: Optional[Path] = None,
                 run_name: Optional[str] = None, resume: Optional[Path] = None,
                 eval_batches: Optional[int] = None, log_every: int = 2000,
                 log=print):
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if h5_dir is not None:
        train_ds = build_h5_dataset(h5_dir, scheme, fold, "train", all_ids=dummy)
        val_ds = build_h5_dataset(h5_dir, scheme, fold, "val", all_ids=dummy)
    else:
        train_ds = build_dataset(scheme, fold, "train", dummy=dummy, af_root=af_root)
        val_ds = build_dataset(scheme, fold, "val", dummy=dummy, af_root=af_root)
    train_loader = make_dataloader(train_ds, bs, shuffle=True,
                                   num_workers=num_workers)
    val_loader = make_dataloader(val_ds, bs, shuffle=False,
                                 num_workers=num_workers)

    model = DistillModel(variant, device=device).to(device)
    loss_mod = DistillLoss(*lambdas).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr,
                                  weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp)

    import datetime
    config = {
        "variant": variant, "scheme": scheme, "fold": fold, "dummy": dummy,
        "lambdas": tuple(lambdas), "bs": bs, "lr": lr, "epochs": epochs,
        "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
        "seed": seed, "num_workers": num_workers, "device": str(device),
        "h5_dir": str(h5_dir) if h5_dir else None,
        "af_root": str(af_root) if af_root else None,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    run_dir = None
    if ckpt_dir is not None:
        run_name = run_name or f"variant{variant}_{scheme}_fold{fold}"
        config["run_name"] = run_name
        run_dir = Path(ckpt_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history: List[dict] = []
    start_epoch, best = 1, float("inf")
    if resume is not None and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history = ckpt.get("history", [])
        best = ckpt.get("best", float("inf"))
        start_epoch = int(ckpt["epoch"]) + 1
        log(f"[train] resumed from {resume} at epoch {start_epoch} (best val "
            f"Cα-RMSD {best:.3f})")

    log(f"[train] variant={variant} scheme={scheme} fold={fold} dummy={dummy} "
        f"device={device} | train={len(train_ds)} val={len(val_ds)} "
        f"bs={bs} epochs={epochs} steps={total_steps} amp={amp}"
        + (f" | ckpt={run_dir}" if run_dir else ""))
    for epoch in range(start_epoch, epochs + 1):
        tr = train_one_epoch(model, train_loader, loss_mod, optimizer,
                             scheduler, device, scaler if amp else None, grad_clip,
                             log=log, log_every=log_every, epoch=epoch)
        ev = evaluate(model, val_loader, loss_mod, device, max_batches=eval_batches)
        history.append({"epoch": epoch, "train": tr, "val": ev})
        log(f"[train] epoch {epoch:3d} | train total {tr['total']:.3f} "
            f"(fape {tr['fape']:.3f} plddt {tr['plddt_ce']:.3f} "
            f"pae {tr['pae_ce']:.3f}) | val total {ev['total']:.3f} "
            f"Cα-RMSD {ev['ca_rmsd']:.2f} pLDDT-sp {ev['plddt_spearman']:.2f} "
            f"PAE-MAE {ev['pae_mae']:.2f}")
        if run_dir is not None:
            if ev["ca_rmsd"] < best:
                best = ev["ca_rmsd"]
                torch.save({"epoch": epoch, "encoder": model.encoder.state_dict(),
                            "val": ev, "config": config}, run_dir / "best.pt")
                log(f"[train]   new best val Cα-RMSD {best:.3f} -> best.pt")
            torch.save({"epoch": epoch, "encoder": model.encoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "history": history,
                        "best": best, "config": config}, run_dir / "last.pt")
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
                           mhc_type) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Parse one teacher example into compact numpy arrays + metadata."""
    ex = parse_example(pdb, peptide, mhc_seq, anchors, mhc_type,
                       return_backbone=True)
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
                     log=print) -> Dict[str, int]:
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
                        r["anchors"], r["mhc_type"])
                except (FileNotFoundError, ValueError) as exc:
                    n_fail += 1
                    pbar.set_postfix(ok=n_ok, missing=n_missing, failed=n_fail)
                    tqdm.write(f"    [skip] {aid}: {str(exc)[:80]}")
                    continue
                g = h5.create_group(aid)
                for k in _H5_DSETS:
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
                      log=print) -> None:
    """Preprocess all (or selected) chunk dirs into HDF5 shards + index. Each
    chunk is independent -> safe to run in parallel jobs (pass disjoint
    --chunks); ``merge`` concatenates per-chunk indices into index.parquet."""
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
                         compression, clevel, log=log)
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
