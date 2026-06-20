"""
H5 store -> PyG graphs for model_2, plus the MHC template pool (point #2).

Each example becomes a ``torch_geometric.data.Data`` with node coords ``pos`` and
features; PyG's DataLoader batches variable-N graphs (no padding). The
``MHCTemplatePool`` lets the MHC be initialised from a *same-length* real MHC
structure (different allele) instead of pure noise — a structural prior the
denoiser then refines toward the target sequence (induced fit).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "model"))
import utils as m1                                  # noqa: E402

N_AATYPE = m1.N_AATYPE
N_SEGMENTS = m1.N_SEGMENTS + 1


def _peptide_mask(segment_id: torch.Tensor) -> torch.Tensor:
    """Peptide = highest segment id in this (unpadded) example."""
    return (segment_id == segment_id.max()).float()


def example_to_data(ex: Dict[str, torch.Tensor]) -> Data:
    seg = ex["segment_id"].long()
    d = Data(
        pos=ex["teacher_ca"].float(),
        aatype=ex["aatype"].long(),
        segment_id=seg,
        residue_index=ex["residue_index"].long(),
        anchor=ex["anchor"].float(),
        pep=_peptide_mask(seg),
        teacher_plddt=ex["teacher_plddt"].float(),
        n_mhc=int(ex["n_mhc"]),
        n_pep=int(ex["n_pep"]),
    )
    d.num_nodes = d.pos.shape[0]
    if "teacher_chi" in ex:                          # side-chain store only
        d.teacher_chi = ex["teacher_chi"].float()           # [N,4,2]
        d.teacher_chi_mask = ex["teacher_chi_mask"].float()  # [N,4]
    return d


class H5GraphDataset(Dataset):
    """Wraps model_1's H5DistillDataset; yields PyG Data objects."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return example_to_data(self.base[i])


def build_loaders(h5_dir, scheme, fold, bs, num_workers, dummy=False):
    if dummy:
        train = m1.build_dataset(scheme, fold, "train", dummy=True)
        val = m1.build_dataset(scheme, fold, "val", dummy=True)
    else:
        train = m1.build_h5_dataset(h5_dir, scheme, fold, "train")
        val = m1.build_h5_dataset(h5_dir, scheme, fold, "val")
    tl = PyGLoader(H5GraphDataset(train), batch_size=bs, shuffle=True,
                   num_workers=num_workers)
    vl = PyGLoader(H5GraphDataset(val), batch_size=bs, shuffle=False,
                   num_workers=num_workers)
    return tl, vl


class MHCTemplatePool:
    """Index of MHC structures grouped by MHC length, for template init. Built
    from the same H5 dataset; fetches a *same-length* template's MHC Cα trace."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.by_len: Dict[int, List[int]] = {}
        # base may be an H5DistillDataset (ids) or a list-style dummy dataset
        for idx in range(len(base_dataset)):
            ex = base_dataset[idx]
            self.by_len.setdefault(int(ex["n_mhc"]), []).append(idx)

    def sample_mhc(self, n_mhc: int, device="cpu") -> Optional[torch.Tensor]:
        """Random same-length MHC Cα [n_mhc,3] (CoM-centred), or None if no match."""
        pool = self.by_len.get(int(n_mhc))
        if not pool:
            return None
        ex = self.base[pool[int(torch.randint(len(pool), (1,)))]]
        ca = ex["teacher_ca"][:int(ex["n_mhc"])].float().to(device)
        return ca - ca.mean(0, keepdim=True)
