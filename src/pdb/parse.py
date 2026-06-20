"""
PDB parser for PMGen-v2 coordinate distillation.

Parses a single PMGen-predicted (teacher) PDB into the per-residue tensors the
encoder + frozen AF2 structure module consume. See module-level conventions:

RESIDUE ORDERING (fixed everywhere): MHC first, then PEPTIDE.
  class I  : [ MHC | peptide ]                       segment_id 0=MHC, 1=peptide
  class II : [ MHC-alpha | MHC-beta | peptide ]      segment_id 0=alpha,1=beta,2=peptide

TEACHER PDB FORMAT (input we parse): a SINGLE chain labelled "A"; MHC residues
first, then peptide, separated by a large (~200, threshold >=150) jump in the
residue NUMBER field (e.g. MHC ...185, peptide 386...). We split on that gap and
validate against the provided sequences.

The amino-acid alphabet is OpenFold's ``residue_constants.restype_order`` (the
structure module's rigid-group constants assume exactly this ordering); unknown
residues map to ``restype_num`` (20).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch


# --- locate the repo root (move-safe) and put OpenFold on the path ----------- #
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "openfold").is_dir() and (parent / "src" / "afbuild").is_dir():
            return parent
    raise RuntimeError(
        "could not locate the PMGen2 repo root (a dir containing both "
        "'openfold/' and 'src/afbuild/') above " + str(Path(__file__).resolve())
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT / "openfold") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "openfold"))

from openfold.np import residue_constants as rc  # noqa: E402

# Biopython is noisy on AF-style PDBs; silence the construction warnings.
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402

GAP_THRESHOLD: int = 150          # residue-number jump that marks a chain break
_UNK: int = rc.restype_num        # aatype value for an unknown residue (20)


# --------------------------------------------------------------------------- #
# Low-level PDB reading
# --------------------------------------------------------------------------- #
def _read_chain_a_calpha(pdb_path: Union[str, Path],
                         collect_atom37: bool = False) -> Dict[str, np.ndarray]:
    """Read chain 'A' residues in file order.

    Returns dict with: ``resseq`` [M] int, ``res1`` [M] '<U1' one-letter codes,
    ``ca`` [M,3], ``n`` [M,3], ``c`` [M,3] float32. With ``collect_atom37`` also
    returns ``atom37`` [M,37,3] (all heavy atoms in OpenFold atom37 order) and
    ``atom37_mask`` [M,37] — needed to derive side-chain torsions.
    Raises on missing chain / missing Cα / insertion codes.
    """
    pdb_path = Path(pdb_path)
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    model = next(iter(structure))                 # first model
    if "A" not in model:
        raise ValueError(f"{pdb_path}: no chain 'A' (chains: "
                         f"{[c.id for c in model]})")
    chain = model["A"]

    nan3 = np.full(3, np.nan, dtype=np.float32)
    resseq: List[int] = []
    res1: List[str] = []
    ca: List[np.ndarray] = []
    bb_n: List[np.ndarray] = []
    bb_c: List[np.ndarray] = []
    atom37: List[np.ndarray] = []
    atom37_mask: List[np.ndarray] = []
    for res in chain.get_residues():
        hetflag, seqid, icode = res.get_id()
        if hetflag.strip():                       # skip HETATM / waters
            continue
        if icode.strip():
            raise ValueError(f"{pdb_path}: unexpected insertion code at residue "
                             f"{seqid}{icode!r}; not supported")
        if "CA" not in res:
            raise ValueError(f"{pdb_path}: residue {res.get_resname()} {seqid} "
                             f"has no Cα atom")
        resseq.append(int(seqid))
        res1.append(rc.restype_3to1.get(res.get_resname(), "X"))
        ca.append(res["CA"].get_coord().astype(np.float32))
        bb_n.append(res["N"].get_coord().astype(np.float32)
                    if "N" in res else nan3)
        bb_c.append(res["C"].get_coord().astype(np.float32)
                    if "C" in res else nan3)
        if collect_atom37:
            pos = np.zeros((rc.atom_type_num, 3), dtype=np.float32)
            msk = np.zeros(rc.atom_type_num, dtype=np.float32)
            for atom in res.get_atoms():
                idx = rc.atom_order.get(atom.get_name())   # heavy atoms only
                if idx is not None:
                    pos[idx] = atom.get_coord().astype(np.float32)
                    msk[idx] = 1.0
            atom37.append(pos)
            atom37_mask.append(msk)

    if not resseq:
        raise ValueError(f"{pdb_path}: chain 'A' has no standard residues")
    out = {
        "resseq": np.asarray(resseq, dtype=np.int64),
        "res1": np.asarray(res1, dtype="<U1"),
        "ca": np.asarray(ca, dtype=np.float32),
        "n": np.asarray(bb_n, dtype=np.float32),
        "c": np.asarray(bb_c, dtype=np.float32),
    }
    if collect_atom37:
        out["atom37"] = np.asarray(atom37, dtype=np.float32)          # [M,37,3]
        out["atom37_mask"] = np.asarray(atom37_mask, dtype=np.float32)  # [M,37]
    return out


def _sidechain_torsions(aatype: np.ndarray, atom37: np.ndarray,
                        atom37_mask: np.ndarray):
    """χ1..χ4 (sin/cos) + mask from atom37, via OpenFold (fp64 for stability).
    Returns (chi_sin_cos [N,4,2] float32, chi_mask [N,4] float32)."""
    from openfold.data.data_transforms import atom37_to_torsion_angles
    protein = {
        "aatype": torch.from_numpy(aatype).long()[None],                 # [1,N]
        "all_atom_positions": torch.from_numpy(atom37).double()[None],   # [1,N,37,3]
        "all_atom_mask": torch.from_numpy(atom37_mask).double()[None],   # [1,N,37]
    }
    out = atom37_to_torsion_angles()(protein)        # @curry1-decorated transform
    # order: [omega, phi, psi, chi1..chi4]; keep the last 4 (side-chain χ)
    chi = out["torsion_angles_sin_cos"][0, :, 3:, :].float().numpy()     # [N,4,2]
    mask = out["torsion_angles_mask"][0, :, 3:].float().numpy()          # [N,4]
    return chi, mask


def _segment_bounds(resseq: np.ndarray, n_expected_segments: int,
                    pdb_path: Union[str, Path]) -> List[slice]:
    """Split an ordered residue-number array into segments at large gaps.

    Expects exactly ``n_expected_segments - 1`` jumps of >= GAP_THRESHOLD, with
    no spurious internal gaps. Returns one slice per segment (in order).
    """
    diffs = np.diff(resseq)
    gap_idx = np.where(diffs >= GAP_THRESHOLD)[0]      # i: gap between i and i+1
    n_gaps = len(gap_idx)
    if n_gaps != n_expected_segments - 1:
        raise ValueError(
            f"{pdb_path}: expected {n_expected_segments - 1} chain-break gap(s) "
            f">= {GAP_THRESHOLD}, found {n_gaps} at residue numbers "
            f"{[(int(resseq[i]), int(resseq[i + 1])) for i in gap_idx]}"
        )
    bounds = [0, *(int(i) + 1 for i in gap_idx), len(resseq)]
    return [slice(bounds[k], bounds[k + 1]) for k in range(n_expected_segments)]


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #
def _parse_anchor_positions(anchors: Union[str, None], n_pep: int,
                            pdb_path: Union[str, Path]) -> List[int]:
    """Parse the ';'-separated, 1-indexed peptide anchor field into 1-indexed
    ints. Empty/NaN -> [] (with a warning). Validates the 1..n_pep range."""
    if anchors is None:
        text = ""
    else:
        text = str(anchors).strip()
    if text == "" or text.lower() == "nan":
        warnings.warn(f"{pdb_path}: empty anchors field -> all-zero anchor vector")
        return []
    try:
        positions = [int(tok) for tok in text.split(";") if tok.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"{pdb_path}: cannot parse anchors {anchors!r}") from exc
    for p in positions:
        if not (1 <= p <= n_pep):
            raise ValueError(
                f"{pdb_path}: anchor position {p} (1-indexed) out of range "
                f"1..{n_pep} for a {n_pep}-mer peptide (anchors={anchors!r})"
            )
    return positions


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_example(
    pdb_path: Union[str, Path],
    peptide_seq: str,
    mhc_seqs: Union[str, Sequence[str]],
    anchors: Union[str, None],
    mhc_type: Union[int, str],
    return_backbone: bool = False,
    return_sidechain: bool = False,
) -> Dict[str, Union[torch.Tensor, int]]:
    """Parse one teacher PDB + its sequences into distillation tensors.

    Parameters
    ----------
    pdb_path        : single-chain-"A" PMGen-predicted PDB.
    peptide_seq     : peptide one-letter sequence.
    mhc_seqs        : class I -> the MHC sequence (str); class II -> (alpha, beta).
    anchors         : ';'-separated 1-indexed peptide anchor positions, e.g. "1;7".
    mhc_type        : 1 (class I) or 2 (class II).
    return_backbone : if True, also return ``teacher_bb`` [N,3,3] = (N, Cα, C)
                      backbone coordinates (for building FAPE target frames).

    Returns a dict of CPU tensors at the example's true length N (no padding):
      aatype [N] long, residue_index [N] long, seq_mask [N] float,
      anchor [N] float, teacher_ca [N,3] float, segment_id [N] long,
      n_mhc int, n_pep int  [, teacher_bb [N,3,3] float].
    """
    mhc_type = int(mhc_type)
    mhc_list: List[str] = [mhc_seqs] if isinstance(mhc_seqs, str) else list(mhc_seqs)

    if mhc_type == 1:
        if len(mhc_list) != 1:
            raise ValueError(f"class I expects 1 MHC sequence, got {len(mhc_list)}")
        seg_seqs = [mhc_list[0], peptide_seq]            # MHC, peptide
        seg_labels = [0, 1]                              # 0=MHC, 1=peptide
    elif mhc_type == 2:
        # Guarded: the test set is class I only and the single-chain "A" splitting
        # convention for alpha/beta has not been verified against real data.
        raise NotImplementedError(
            "class II parsing is intentionally guarded: confirm the teacher "
            "single-chain-'A' layout for MHC-alpha/beta (one ~200 gap or two?) "
            "against real data before enabling. The segment machinery here "
            "generalises to seg_seqs=[alpha, beta, peptide], seg_labels=[0,1,2]."
        )
    else:
        raise ValueError(f"unsupported mhc_type {mhc_type!r} (expected 1 or 2)")

    rec = _read_chain_a_calpha(pdb_path, collect_atom37=return_sidechain)
    n_segments = len(seg_seqs)
    slices = _segment_bounds(rec["resseq"], n_segments, pdb_path)

    # ---- verify each segment against the provided sequence (len + identity) ---
    for label_seq, sl in zip(seg_seqs, slices):
        pdb_seg = "".join(rec["res1"][sl].tolist())
        if len(pdb_seg) != len(label_seq):
            raise ValueError(
                f"{pdb_path}: segment length mismatch — PDB {len(pdb_seg)} vs "
                f"sequence {len(label_seq)} (segment '{label_seq[:8]}...')"
            )
        if pdb_seg != label_seq:
            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(pdb_seg, label_seq))
                     if a != b]
            raise ValueError(
                f"{pdb_path}: residue identity mismatch vs provided sequence at "
                f"{len(diffs)} position(s); first: index {diffs[0][0]} "
                f"PDB={diffs[0][1]} seq={diffs[0][2]}"
            )

    seg_lengths = [sl.stop - sl.start for sl in slices]
    n_pep = seg_lengths[-1]
    n_mhc = int(sum(seg_lengths[:-1]))
    n_total = int(sum(seg_lengths))

    # ---- aatype from the SEQUENCES (mhc... then peptide), via restype_order ---
    full_seq = "".join(seg_seqs)
    aatype = np.fromiter(
        (rc.restype_order.get(aa, _UNK) for aa in full_seq),
        dtype=np.int64, count=n_total,
    )

    # ---- residue_index straight from the PDB numbering (keeps the +200 gap) ---
    residue_index = rec["resseq"].copy()
    boundary = n_mhc                                     # first peptide index
    gap = int(residue_index[boundary] - residue_index[boundary - 1])
    if gap < GAP_THRESHOLD:
        raise ValueError(
            f"{pdb_path}: expected MHC->peptide numbering gap >= {GAP_THRESHOLD} "
            f"at index {boundary}, found {gap}"
        )

    # ---- segment_id ----
    segment_id = np.empty(n_total, dtype=np.int64)
    for label, sl in zip(seg_labels, slices):
        segment_id[sl] = label

    # ---- anchor: 1s at peptide anchor positions -> full index n_mhc+(pos-1) ----
    anchor = np.zeros(n_total, dtype=np.float32)
    for pos in _parse_anchor_positions(anchors, n_pep, pdb_path):
        anchor[n_mhc + (pos - 1)] = 1.0

    teacher_ca = rec["ca"]
    if teacher_ca.shape != (n_total, 3):
        raise ValueError(f"{pdb_path}: Cα shape {teacher_ca.shape} != ({n_total}, 3)")
    if not np.isfinite(teacher_ca).all():
        raise ValueError(f"{pdb_path}: non-finite Cα coordinates")

    out: Dict[str, Union[torch.Tensor, int]] = {
        "aatype": torch.from_numpy(aatype),
        "residue_index": torch.from_numpy(residue_index),
        "seq_mask": torch.ones(n_total, dtype=torch.float32),
        "anchor": torch.from_numpy(anchor),
        "teacher_ca": torch.from_numpy(teacher_ca),
        "segment_id": torch.from_numpy(segment_id),
        "n_mhc": n_mhc,
        "n_pep": int(n_pep),
    }
    if return_backbone:
        teacher_bb = np.stack([rec["n"], rec["ca"], rec["c"]], axis=1)  # [N,3,3]
        if not np.isfinite(teacher_bb).all():
            raise ValueError(f"{pdb_path}: missing/non-finite backbone N or C "
                             "atom(s) required for FAPE frames")
        out["teacher_bb"] = torch.from_numpy(teacher_bb)
    if return_sidechain:
        chi, chi_mask = _sidechain_torsions(aatype, rec["atom37"],
                                            rec["atom37_mask"])
        if chi.shape != (n_total, 4, 2):
            raise ValueError(f"{pdb_path}: chi shape {chi.shape} != ({n_total},4,2)")
        out["teacher_chi"] = torch.from_numpy(chi)               # [N,4,2] sin/cos
        out["teacher_chi_mask"] = torch.from_numpy(chi_mask)     # [N,4]
    return out


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def collate_fn(
    examples: Sequence[Dict[str, Union[torch.Tensor, int]]],
) -> Dict[str, torch.Tensor]:
    """Pad a list of ``parse_example`` outputs to max-N for batched training.

    Padding positions get ``seq_mask`` = 0 (so the frozen SM / loss ignore them),
    ``aatype`` = 0, ``residue_index`` = 0, ``anchor`` = 0, ``teacher_ca`` = 0 and
    ``segment_id`` = -1 (a sentinel that is never a real segment). Returns a dict
    with batched tensors plus per-example ``n_mhc``/``n_pep``/``length`` [B].
    """
    if not examples:
        raise ValueError("collate_fn received an empty list")
    batch = len(examples)
    lengths = torch.tensor([int(e["aatype"].shape[0]) for e in examples],
                           dtype=torch.long)
    max_n = int(lengths.max())

    out = {
        "aatype": torch.zeros(batch, max_n, dtype=torch.long),
        "residue_index": torch.zeros(batch, max_n, dtype=torch.long),
        "seq_mask": torch.zeros(batch, max_n, dtype=torch.float32),
        "anchor": torch.zeros(batch, max_n, dtype=torch.float32),
        "teacher_ca": torch.zeros(batch, max_n, 3, dtype=torch.float32),
        "segment_id": torch.full((batch, max_n), -1, dtype=torch.long),
        "n_mhc": torch.tensor([int(e["n_mhc"]) for e in examples], dtype=torch.long),
        "n_pep": torch.tensor([int(e["n_pep"]) for e in examples], dtype=torch.long),
        "length": lengths,
    }
    for i, e in enumerate(examples):
        n = int(e["aatype"].shape[0])
        out["aatype"][i, :n] = e["aatype"]
        out["residue_index"][i, :n] = e["residue_index"]
        out["seq_mask"][i, :n] = e["seq_mask"]
        out["anchor"][i, :n] = e["anchor"]
        out["teacher_ca"][i, :n] = e["teacher_ca"]
        out["segment_id"][i, :n] = e["segment_id"]
    return out
