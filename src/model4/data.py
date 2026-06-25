"""
model4 / plddt_model — AFDB monomer dataset (stage-1 pretraining).

Reads the flattened ``plddt_dataset.h5`` (2.3M proteins) and yields fixed-length
256-residue crops with: shifted tokens (+1 so 0=pad), per-residue pLDDT target,
domain-derived segment ids with inserted SEP tokens (domains simulate chains),
and dense pair features encoding the *anchor* curriculum:

  - anchor_state  [L,L] in {0=Unknown, 1=NoAnchor, 2=AnchorKnown}
  - pe_bin        [L,L] in 0..15 (0..13 contact-distance buckets, 14 non-contact,
                                   15 = null / no-PE)
  - anchor_target [L,L] 1.0 for anchor contacts (<8A & cross-segment or |i-j|>K)
  - anchor_loss_mask [L,L] real-pair mask with revealed pairs excluded

Anchor reveal curriculum (per example): with prob (1-reveal_prob) reveal nothing;
otherwise reveal U(0, reveal_frac_max) of the anchors as AnchorKnown (half also get
the distance-bin PE), plus a small fraction of true non-contacts as NoAnchor.

H5 schema (verified): flattened arrays addressed by per-protein pointers
``seq_offset/seq_length`` (sequence uint8 0=A..24=U; plddt float16 0-100) and
``dist_offset/dist_count`` (sparse <8A CA-CA contacts: dist_i/dist_j local i<j,
dist_value float16).

Env: pmgen2.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model import (PAD_TOKEN, SEP_TOKEN, MASK_TOKEN, N_AA, VOCAB_SIZE,
                   MAX_SEGMENTS, N_PE_BINS, PE_NULL_BIN)

CONTACT_CUTOFF = 8.0
N_CONTACT_BINS = 14            # PE bins 0..13 over [0, 8) Angstrom
NONCONTACT_BIN = 14           # explicit "not in proximity"


def _dist_bin(d: float) -> int:
    b = int(d / CONTACT_CUTOFF * N_CONTACT_BINS)
    return min(N_CONTACT_BINS - 1, max(0, b))


class AFDBMonomerDataset(Dataset):
    def __init__(self, h5_path: str, domains_tsv: Optional[str] = None, *,
                 split: str = "train", crop_len: int = 256, max_offset: int = 32,
                 anchor_seq_sep: int = 6, reveal_prob: float = 0.2,
                 reveal_frac_max: float = 0.5, pe_reveal_frac: float = 0.5,
                 neg_reveal_frac: float = 0.02, mlm_frac: float = 0.15,
                 val_frac: float = 0.005, seed: int = 0, min_len: int = 8,
                 limit: Optional[int] = None):
        self.h5_path = str(h5_path)
        self.crop_len = crop_len
        self.max_offset = max_offset
        self.K = anchor_seq_sep
        self.reveal_prob = reveal_prob
        self.reveal_frac_max = reveal_frac_max
        self.pe_reveal_frac = pe_reveal_frac
        self.neg_reveal_frac = neg_reveal_frac
        self.mlm_frac = mlm_frac
        self._h5: Optional[h5py.File] = None

        # pointer arrays into memory (small: ~18MB each at most)
        with h5py.File(self.h5_path, "r") as f:
            self.seq_offset = f["seq_offset"][:]
            self.seq_length = f["seq_length"][:]
            self.dist_offset = f["dist_offset"][:]
            self.dist_count = f["dist_count"][:]
            self.pdb_id = f["pdb_id"][:]                       # bytes array
        n = self.seq_length.shape[0]

        # deterministic train/val split (no index file needed)
        perm = np.random.default_rng(seed).permutation(n)
        n_val = int(n * val_frac)
        idx = perm[:n_val] if split == "val" else perm[n_val:]
        idx = idx[self.seq_length[idx] >= min_len]
        if limit is not None:
            idx = idx[:limit]
        self.indices = idx

        self.domains = self._load_domains(domains_tsv) if domains_tsv else {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_domains(tsv: str) -> Dict[str, List[Tuple[int, int]]]:
        df = pd.read_csv(tsv, sep="\t", usecols=["mapped_queries", "domain_start", "domain_end"])
        df = df.dropna(subset=["mapped_queries"])
        key = df["mapped_queries"].astype(str).str.replace(r"\.pdb$", "", regex=True)
        out: Dict[str, List[Tuple[int, int]]] = {}
        for k, s, e in zip(key.values, df["domain_start"].values, df["domain_end"].values):
            out.setdefault(k, []).append((int(s), int(e)))
        return out

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------ #
    def _segment_ids(self, key: str, sl: int) -> np.ndarray:
        doms = self.domains.get(key)
        if not doms:
            return np.zeros(sl, dtype=np.int64)
        starts = np.array(sorted(s for s, _ in doms))                  # 1-indexed
        res = np.arange(1, sl + 1)
        # segment = number of domain-starts at or before this residue
        return np.searchsorted(starts, res, side="right").astype(np.int64)

    def _augment(self, tok: np.ndarray, seg: np.ndarray):
        """Insert SEP between segments; return augmented arrays + posmap(orig->aug)."""
        sl = tok.shape[0]
        gap = self.max_offset + 1
        a_tok, a_seg, a_real, a_true, a_ri = [], [], [], [], []
        posmap = np.empty(sl, dtype=np.int64)
        ri = 0
        for p in range(sl):
            if p > 0 and seg[p] != seg[p - 1]:
                a_tok.append(SEP_TOKEN); a_seg.append(seg[p]); a_real.append(False)
                a_true.append(-1); a_ri.append(ri); ri += gap
            posmap[p] = len(a_tok)
            a_tok.append(int(tok[p])); a_seg.append(int(seg[p])); a_real.append(True)
            a_true.append(p); a_ri.append(ri); ri += 1
        return (np.array(a_tok, np.int64), np.array(a_seg, np.int64),
                np.array(a_real, bool), np.array(a_true, np.int64),
                np.array(a_ri, np.int64), posmap)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        f = self._file()
        idx = int(self.indices[i])
        so, sl = int(self.seq_offset[idx]), int(self.seq_length[idx])
        tok_raw = f["sequence"][so:so + sl].astype(np.int64) + 1        # shift: 0=pad
        plddt = f["plddt"][so:so + sl].astype(np.float32) / 100.0
        do, dc = int(self.dist_offset[idx]), int(self.dist_count[idx])
        di = f["dist_i"][do:do + dc].astype(np.int64)
        dj = f["dist_j"][do:do + dc].astype(np.int64)
        dv = f["dist_value"][do:do + dc].astype(np.float32)
        key = self.pdb_id[idx].decode().strip()
        seg = self._segment_ids(key, sl)

        # augment with separators, then crop a fixed window over augmented coords
        a_tok, a_seg, a_real, a_true, a_ri, posmap = self._augment(tok_raw, seg)
        La = a_tok.shape[0]
        L = self.crop_len
        w0 = random.randint(0, La - L) if La > L else 0
        w1 = min(w0 + L, La)
        Lc = w1 - w0

        # map original-residue arrays (plddt, true_pos) into cropped positions
        c_tok = a_tok[w0:w1]; c_seg = a_seg[w0:w1]; c_real = a_real[w0:w1]
        c_true = a_true[w0:w1]; c_ri = a_ri[w0:w1] - a_ri[w0]
        # renumber segments within crop to 0..k
        uniq = {s: n for n, s in enumerate(sorted(set(c_seg.tolist())))}
        c_seg = np.array([uniq[s] for s in c_seg.tolist()], np.int64)

        # per-cropped pLDDT (only real positions carry a target)
        c_plddt = np.zeros(Lc, np.float32)
        real_idx = np.where(c_real)[0]
        # map cropped real position -> original residue (a_true) -> plddt
        c_plddt[real_idx] = plddt[c_true[real_idx]]

        # edges within the crop window (orig residue -> aug pos -> cropped pos)
        ai = posmap[di]; aj = posmap[dj]
        em = (ai >= w0) & (ai < w1) & (aj >= w0) & (aj < w1)
        ci = (ai[em] - w0); cj = (aj[em] - w0); cd = dv[em]

        out = self._build_example(L, Lc, c_tok, c_seg, c_real, c_true, c_ri,
                                  c_plddt, ci, cj, cd)
        return out

    # ------------------------------------------------------------------ #
    def _build_example(self, L, Lc, c_tok, c_seg, c_real, c_true, c_ri,
                       c_plddt, ci, cj, cd) -> Dict[str, torch.Tensor]:
        tokens = np.zeros(L, np.int64)
        segment_id = np.zeros(L, np.int64)
        residue_index = np.zeros(L, np.int64)
        plddt = np.zeros(L, np.float32)
        residue_mask = np.zeros(L, np.float32)
        token_mask = np.zeros(L, np.float32)
        tokens[:Lc] = c_tok
        segment_id[:Lc] = c_seg
        residue_index[:Lc] = c_ri
        plddt[:Lc] = c_plddt
        residue_mask[:Lc] = c_real.astype(np.float32)     # real AA only
        token_mask[:Lc] = (c_tok != PAD_TOKEN).astype(np.float32)  # AA + SEP

        anchor_state = np.zeros((L, L), np.int64)
        pe_bin = np.full((L, L), PE_NULL_BIN, np.int64)
        anchor_target = np.zeros((L, L), np.float32)
        is_contact = np.zeros((L, L), bool)

        # classify contacts -> anchors (cross-segment OR |orig_i - orig_j| > K)
        anchors = []                                       # (a, b, dist)
        for a, b, d in zip(ci.tolist(), cj.tolist(), cd.tolist()):
            is_contact[a, b] = is_contact[b, a] = True
            cross = c_seg[a] != c_seg[b]
            seqsep = abs(int(c_true[a]) - int(c_true[b])) > self.K
            if cross or seqsep:
                anchor_target[a, b] = anchor_target[b, a] = 1.0
                anchors.append((a, b, d))

        # real-pair loss mask (both real, i != j); revealed pairs removed below
        rm = residue_mask[:, None] * residue_mask[None, :]
        np.fill_diagonal(rm, 0.0)
        anchor_loss_mask = rm.copy()

        # ---- reveal curriculum ----
        if random.random() < self.reveal_prob and anchors:
            frac = random.uniform(0.0, self.reveal_frac_max)
            n_rev = int(round(frac * len(anchors)))
            for a, b, d in random.sample(anchors, min(n_rev, len(anchors))):
                anchor_state[a, b] = anchor_state[b, a] = 2            # AnchorKnown
                anchor_loss_mask[a, b] = anchor_loss_mask[b, a] = 0.0
                if random.random() < self.pe_reveal_frac:
                    bb = _dist_bin(d)
                    pe_bin[a, b] = pe_bin[b, a] = bb
            # reveal a small fraction of true non-contacts as NoAnchor
            real_pos = np.where(c_real)[0]
            if len(real_pos) > 1:
                n_neg = min(200, int(round(self.neg_reveal_frac * Lc * Lc)))
                tries = 0
                placed = 0
                while placed < n_neg and tries < 8 * n_neg:
                    tries += 1
                    a, b = np.random.choice(real_pos, 2, replace=False)
                    if is_contact[a, b] or anchor_state[a, b] != 0:
                        continue
                    anchor_state[a, b] = anchor_state[b, a] = 1        # NoAnchor
                    anchor_loss_mask[a, b] = anchor_loss_mask[b, a] = 0.0
                    placed += 1

        # ---- MLM (BERT 80/10/10) on real AA positions ----
        mlm_labels = np.full(L, -100, np.int64)
        real_pos = np.where(c_real)[0]
        if len(real_pos) > 0 and self.mlm_frac > 0:
            n_mask = max(1, int(round(self.mlm_frac * len(real_pos))))
            for p in np.random.choice(real_pos, min(n_mask, len(real_pos)), replace=False):
                mlm_labels[p] = tokens[p]
                r = random.random()
                if r < 0.8:
                    tokens[p] = MASK_TOKEN
                elif r < 0.9:
                    tokens[p] = random.randint(1, N_AA)               # random AA 1..25

        return {
            "tokens": torch.from_numpy(tokens),
            "mlm_labels": torch.from_numpy(mlm_labels),
            "plddt": torch.from_numpy(plddt),
            "residue_mask": torch.from_numpy(residue_mask),
            "token_mask": torch.from_numpy(token_mask),
            "segment_id": torch.from_numpy(segment_id),
            "residue_index": torch.from_numpy(residue_index),
            "anchor_state": torch.from_numpy(anchor_state),
            "pe_bin": torch.from_numpy(pe_bin),
            "anchor_target": torch.from_numpy(anchor_target),
            "anchor_loss_mask": torch.from_numpy(anchor_loss_mask),
        }


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)
