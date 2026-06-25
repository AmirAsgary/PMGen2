"""
model4 / plddt_model — AFDB monomer dataset (stage-1 pretraining).

Yields VARIABLE-length protein crops (no fixed pad) so a length-bucketing batch
sampler can pack similar-length proteins and pad only to the per-batch max — most
AFDB chains are shorter than the crop, so this removes ~40% wasted N^2 pair compute.

Each example carries: shifted tokens (+1, 0=pad), per-residue pLDDT target,
domain-derived segment ids with inserted SEP tokens (domains simulate chains),
per-segment residue_index (so cross-chain pairs are distinguishable), and dense
uint8 pair features for the anchor curriculum + a distogram target:

  anchor_state  [L,L]  {0=Unknown, 1=NoAnchor, 2=AnchorKnown}
  pe_bin        [L,L]  0..15 (0..13 contact-dist buckets, 14 non-contact, 15 null)
  anchor_target [L,L]  1 for anchor contacts (<8A & cross-segment or |i-j|>K)
  disto_target  [L,L]  0..15 (15 buckets over [0,8A) + bin 15 = ">=8A")  -> dense, full supervision
  *_loss_mask   [L,L]  real-pair masks with revealed/leaked pairs excluded

Anchor reveal curriculum (per example): with prob (1-reveal_prob) reveal nothing;
else reveal U(0, reveal_frac_max) of anchors as AnchorKnown (half also get the
distance-bin PE; those are dropped from the distogram loss to avoid leakage), plus
a small fraction of true non-contacts as NoAnchor.

Env: pmgen2.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model import (PAD_TOKEN, SEP_TOKEN, MASK_TOKEN, N_AA, SEG_SEP,
                   PE_NULL_BIN, N_DIST_FAR)

CONTACT_CUTOFF = 8.0
N_PE_CONTACT_BINS = 14         # PE bins 0..13 over [0, 8) Angstrom
N_DIST_SUBBINS = 15            # distogram buckets over [0, 8) Angstrom (bin 15 = >=8A)


def _pe_bin(d: float) -> int:
    return min(N_PE_CONTACT_BINS - 1, max(0, int(d / CONTACT_CUTOFF * N_PE_CONTACT_BINS)))


class AFDBMonomerDataset(Dataset):
    def __init__(self, h5_path: str, domains_tsv: Optional[str] = None, *,
                 split: str = "train", crop_len: int = 256, max_offset: int = 32,
                 anchor_seq_sep: int = 6, reveal_prob: float = 0.2,
                 reveal_frac_max: float = 0.5, pe_reveal_frac: float = 0.5,
                 neg_reveal_frac: float = 0.02, mlm_frac: float = 0.15,
                 val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
                 min_len: int = 8, limit: Optional[int] = None):
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

        with h5py.File(self.h5_path, "r") as f:
            self.seq_offset = f["seq_offset"][:]
            self.seq_length = f["seq_length"][:]
            self.dist_offset = f["dist_offset"][:]
            self.dist_count = f["dist_count"][:]
            self.pdb_id = f["pdb_id"][:]
        n = self.seq_length.shape[0]

        # deterministic random 80/10/10 train/val/test split
        assert split in ("train", "val", "test"), split
        perm = np.random.default_rng(seed).permutation(n)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        parts = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val],
                 "train": perm[n_test + n_val:]}
        idx = parts[split]
        idx = idx[self.seq_length[idx] >= min_len]
        if limit is not None:
            idx = idx[:limit]
        self.indices = idx
        # bucketing key (real example length is ~ this + a few SEP tokens)
        self.lengths = np.minimum(self.seq_length[idx], crop_len).astype(np.int64)

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

    def _segment_ids(self, key: str, sl: int) -> np.ndarray:
        doms = self.domains.get(key)
        if not doms:
            return np.zeros(sl, dtype=np.int64)
        starts = np.array(sorted(s for s, _ in doms))                  # 1-indexed
        res = np.arange(1, sl + 1)
        return np.searchsorted(starts, res, side="right").astype(np.int64)

    # ------------------------------------------------------------------ #
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        f = self._file()
        idx = int(self.indices[i])
        so, sl = int(self.seq_offset[idx]), int(self.seq_length[idx])
        # --- crop on ORIGINAL residues FIRST (cheap; avoids augmenting 2000+ residues) ---
        L = self.crop_len
        o0 = random.randint(0, sl - L) if sl > L else 0
        o1 = min(o0 + L, sl)
        ow = o1 - o0
        tok = f["sequence"][so + o0:so + o1].astype(np.int64) + 1       # 0=pad
        plddt = f["plddt"][so + o0:so + o1].astype(np.float32) / 100.0
        do, dc = int(self.dist_offset[idx]), int(self.dist_count[idx])
        di = f["dist_i"][do:do + dc].astype(np.int64)
        dj = f["dist_j"][do:do + dc].astype(np.int64)
        dv = f["dist_value"][do:do + dc].astype(np.float32)
        em = (di >= o0) & (di < o1) & (dj >= o0) & (dj < o1)
        li, lj, ld = di[em] - o0, dj[em] - o0, dv[em]                   # window-local

        key = self.pdb_id[idx].decode().strip()
        seg = self._segment_ids(key, sl)[o0:o1]
        uniq = {s: n for n, s in enumerate(sorted(set(seg.tolist())))}  # renumber 0..k
        seg = np.array([uniq[s] for s in seg.tolist()], np.int64)

        return self._augment_and_build(tok, seg, plddt, li, lj, ld, ow)

    def _augment_and_build(self, tok, seg, plddt, li, lj, ld, ow):
        # insert SEP at segment changes; per-segment residue_index; map local->aug
        a_tok, a_seg, a_real, a_loc, a_ri = [], [], [], [], []
        posmap = np.empty(ow, dtype=np.int64)
        ri = 0
        for p in range(ow):
            if p > 0 and seg[p] != seg[p - 1]:
                a_tok.append(SEP_TOKEN); a_seg.append(SEG_SEP); a_real.append(False)
                a_loc.append(-1); a_ri.append(0); ri = 0
            posmap[p] = len(a_tok)
            a_tok.append(int(tok[p])); a_seg.append(int(seg[p])); a_real.append(True)
            a_loc.append(p); a_ri.append(ri); ri += 1
        tokens = np.array(a_tok, np.int64)
        segment_id = np.array(a_seg, np.int64)
        real = np.array(a_real, bool)
        a_loc = np.array(a_loc, np.int64)
        residue_index = np.array(a_ri, np.int64)
        Lc = tokens.shape[0]
        ei, ej = posmap[li], posmap[lj]                                # aug positions of edges

        plddt_full = np.zeros(Lc, np.float32)
        plddt_full[real] = plddt[a_loc[real]]

        # ---- dense pair targets/features (uint8 to cut H2D bandwidth) ----
        anchor_state = np.zeros((Lc, Lc), np.uint8)
        pe_bin = np.full((Lc, Lc), PE_NULL_BIN, np.uint8)
        anchor_target = np.zeros((Lc, Lc), np.uint8)
        disto_target = np.full((Lc, Lc), N_DIST_FAR, np.uint8)         # default ">=8A"
        is_contact = np.zeros((Lc, Lc), bool)
        if ei.size:
            dbin = np.minimum(N_DIST_SUBBINS - 1,
                              (ld / CONTACT_CUTOFF * N_DIST_SUBBINS).astype(np.int64))
            is_contact[ei, ej] = is_contact[ej, ei] = True
            disto_target[ei, ej] = disto_target[ej, ei] = dbin.astype(np.uint8)
            cross = segment_id[ei] != segment_id[ej]
            seqsep = np.abs(a_loc[ei] - a_loc[ej]) > self.K
            isa = cross | seqsep
            aei, aej, aed = ei[isa], ej[isa], ld[isa]
            anchor_target[aei, aej] = anchor_target[aej, aei] = 1
        else:
            aei = aej = aed = np.empty(0, np.int64)

        rm = real.astype(np.float32)
        real_pair = (rm[:, None] * rm[None, :]).astype(np.uint8)
        np.fill_diagonal(real_pair, 0)
        anchor_loss_mask = real_pair.copy()
        disto_loss_mask = real_pair.copy()

        # ---- reveal curriculum ----
        if random.random() < self.reveal_prob and aei.size:
            n_rev = int(round(random.uniform(0.0, self.reveal_frac_max) * aei.size))
            if n_rev > 0:
                sel = np.random.choice(aei.size, min(n_rev, aei.size), replace=False)
                ra, rb, rd = aei[sel], aej[sel], aed[sel]
                anchor_state[ra, rb] = anchor_state[rb, ra] = 2        # AnchorKnown
                anchor_loss_mask[ra, rb] = anchor_loss_mask[rb, ra] = 0
                pe_mask = np.random.random(sel.size) < self.pe_reveal_frac
                if pe_mask.any():
                    pa, pb, pd = ra[pe_mask], rb[pe_mask], rd[pe_mask]
                    bb = np.array([_pe_bin(float(x)) for x in pd], np.uint8)
                    pe_bin[pa, pb] = pe_bin[pb, pa] = bb
                    disto_loss_mask[pa, pb] = disto_loss_mask[pb, pa] = 0  # distance leaked
            # small fraction of true non-contacts revealed as NoAnchor
            real_pos = np.where(real)[0]
            if real_pos.size > 1:
                n_neg = min(200, int(round(self.neg_reveal_frac * Lc * Lc)))
                placed, tries = 0, 0
                while placed < n_neg and tries < 8 * max(1, n_neg):
                    tries += 1
                    a, b = np.random.choice(real_pos, 2, replace=False)
                    if is_contact[a, b] or anchor_state[a, b] != 0:
                        continue
                    anchor_state[a, b] = anchor_state[b, a] = 1         # NoAnchor
                    anchor_loss_mask[a, b] = anchor_loss_mask[b, a] = 0
                    placed += 1

        # ---- MLM (BERT 80/10/10) on real AA positions ----
        mlm_labels = np.full(Lc, -100, np.int64)
        real_pos = np.where(real)[0]
        if real_pos.size and self.mlm_frac > 0:
            n_mask = max(1, int(round(self.mlm_frac * real_pos.size)))
            for p in np.random.choice(real_pos, min(n_mask, real_pos.size), replace=False):
                mlm_labels[p] = tokens[p]
                r = random.random()
                if r < 0.8:
                    tokens[p] = MASK_TOKEN
                elif r < 0.9:
                    tokens[p] = random.randint(1, N_AA)

        token_mask = (tokens != PAD_TOKEN).astype(np.float32)          # AA + SEP
        return {
            "tokens": torch.from_numpy(tokens),
            "mlm_labels": torch.from_numpy(mlm_labels),
            "plddt": torch.from_numpy(plddt_full),
            "residue_mask": torch.from_numpy(rm),                       # real AA only
            "token_mask": torch.from_numpy(token_mask),
            "segment_id": torch.from_numpy(segment_id),
            "residue_index": torch.from_numpy(residue_index),
            "anchor_state": torch.from_numpy(anchor_state),
            "pe_bin": torch.from_numpy(pe_bin),
            "anchor_target": torch.from_numpy(anchor_target),
            "anchor_loss_mask": torch.from_numpy(anchor_loss_mask),
            "disto_target": torch.from_numpy(disto_target),
            "disto_loss_mask": torch.from_numpy(disto_loss_mask),
        }


# --------------------------------------------------------------------------- #
# Variable-length collate: pad to the per-batch max (not a global crop length).
# --------------------------------------------------------------------------- #
_PAD1 = {"tokens": (PAD_TOKEN, torch.long), "mlm_labels": (-100, torch.long),
         "plddt": (0.0, torch.float32), "residue_mask": (0.0, torch.float32),
         "token_mask": (0.0, torch.float32), "segment_id": (SEG_SEP, torch.long),
         "residue_index": (0, torch.long)}
_PAD2 = {"anchor_state": 0, "pe_bin": PE_NULL_BIN, "anchor_target": 0,
         "anchor_loss_mask": 0, "disto_target": 0, "disto_loss_mask": 0}


def collate_varlen(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(samples)
    Nmax = max(s["tokens"].shape[0] for s in samples)
    out: Dict[str, torch.Tensor] = {}
    for k, (fill, dt) in _PAD1.items():
        t = torch.full((B, Nmax), fill, dtype=dt)
        for b, s in enumerate(samples):
            t[b, :s[k].shape[0]] = s[k]
        out[k] = t
    for k, fill in _PAD2.items():
        t = torch.full((B, Nmax, Nmax), fill, dtype=torch.uint8)
        for b, s in enumerate(samples):
            n = s[k].shape[0]
            t[b, :n, :n] = s[k]
        out[k] = t
    return out


# --------------------------------------------------------------------------- #
# Length-bucketing batch sampler (A1): packs similar-length proteins; shards
# batches evenly across DDP ranks so every rank runs the same #optimizer steps.
# --------------------------------------------------------------------------- #
class LengthBucketBatchSampler:
    def __init__(self, lengths, batch_size: int, num_replicas: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0, megabatch_mult: int = 50,
                 drop_last: bool = True):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.world = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.mm = megabatch_mult
        self.drop_last = drop_last
        self.epoch = 0
        self._len = len(self._build(0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build(self, epoch: int):
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        mb = self.bs * self.mm
        batches = []
        for i in range(0, n, mb):
            chunk = order[i:i + mb]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for k in range(0, len(chunk), self.bs):
                b = chunk[k:k + self.bs]
                if len(b) == self.bs or not self.drop_last:
                    batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        nb = (len(batches) // self.world) * self.world                 # even across ranks
        batches = batches[:nb][self.rank::self.world]
        return batches

    def __iter__(self):
        for b in self._build(self.epoch):
            yield b.tolist()

    def __len__(self) -> int:
        return self._len


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)
