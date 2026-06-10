"""
STEP 4 — overfit-one sanity check.

Proves the data pipeline + frozen AF2 stack + superposed-Cα loss can drive
predicted Cα onto the teacher Cα for ONE real example, using a *throwaway*
trivial encoder (NOT the real PMGen-v2 encoder). If the RMSD drops well under
~2 Å this validates indexing / ordering / loss; a high plateau would indicate a
bug there rather than a capacity limit.

Run:
  ~/miniforge3/envs/pmgen2/bin/python src/pdb/overfit_one.py \
      --id 3GSO_0_0 --steps 400
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parse  # same directory
from parse import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src" / "afbuild"))
from utils import load_frozen_fold  # noqa: E402  (afbuild/utils.py)

TEST_DIR = REPO_ROOT / "data" / "test"
N_AATYPE = parse.rc.restype_num + 1          # 21 (20 standard + unknown)
N_SEGMENTS = 3                               # 0=MHC/alpha, 1=beta/peptide, 2=peptide


# --------------------------------------------------------------------------- #
# Throwaway encoder (deliberately trivial; replaced later by the real one)
# --------------------------------------------------------------------------- #
class ThrowawayEncoder(nn.Module):
    """tokens = embed(aatype)+embed(segment)+embed(anchor)+sinusoidal(res_index)
    -> s = Linear(tokens) [N,384];  z = MLP(outer-concat tokens) [N,N,128]."""

    def __init__(self, d_token: int = 192, c_s: int = 384, c_z: int = 128,
                 z_hidden: int = 256) -> None:
        super().__init__()
        self.d_token = d_token
        self.aatype_emb = nn.Embedding(N_AATYPE, d_token)
        self.segment_emb = nn.Embedding(N_SEGMENTS, d_token)
        self.anchor_emb = nn.Embedding(2, d_token)
        self.to_s = nn.Linear(d_token, c_s)
        self.to_z = nn.Sequential(
            nn.Linear(2 * d_token, z_hidden), nn.ReLU(),
            nn.Linear(z_hidden, c_z),
        )

    @staticmethod
    def _sinusoidal(residue_index: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal positional encoding over the (gapped) residue numbering."""
        pos = residue_index.float()[:, None]                       # [N,1]
        i = torch.arange(0, dim, 2, device=pos.device).float()     # [dim/2]
        div = torch.exp(-np.log(10000.0) * i / dim)
        pe = torch.zeros(pos.shape[0], dim, device=pos.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, aatype: torch.Tensor, segment_id: torch.Tensor,
                anchor: torch.Tensor, residue_index: torch.Tensor):
        # all inputs are [N] for the single example
        tokens = (self.aatype_emb(aatype)
                  + self.segment_emb(segment_id)
                  + self.anchor_emb(anchor.long())
                  + self._sinusoidal(residue_index, self.d_token))      # [N,d]
        s = self.to_s(tokens)                                           # [N,384]
        n = tokens.shape[0]
        ti = tokens[:, None, :].expand(n, n, self.d_token)
        tj = tokens[None, :, :].expand(n, n, self.d_token)
        z = self.to_z(torch.cat([ti, tj], dim=-1))                      # [N,N,128]
        return s, z


# --------------------------------------------------------------------------- #
# Superposed-Cα loss (differentiable Kabsch alignment)
# --------------------------------------------------------------------------- #
def kabsch_align(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Rigidly align ``pred`` [N,3] onto ``target`` [N,3] over masked points
    (rotation + translation only, differentiable via SVD). Returns aligned pred."""
    w = mask.float()[:, None]                          # [N,1]
    wsum = w.sum().clamp_min(1.0)
    mu_p = (pred * w).sum(0) / wsum
    mu_t = (target * w).sum(0) / wsum
    p0 = pred - mu_p
    t0 = target - mu_t
    h = (p0 * w).transpose(0, 1) @ t0                  # [3,3] weighted covariance
    u, _, vt = torch.linalg.svd(h)
    v = vt.transpose(0, 1)
    d = torch.sign(torch.linalg.det(v @ u.transpose(0, 1)))
    diag = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    rot = v @ diag @ u.transpose(0, 1)                 # [3,3]
    return p0 @ rot.transpose(0, 1) + mu_t


def superposed_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (smooth_l1 loss on aligned masked Cα, masked Cα RMSD in Å)."""
    aligned = kabsch_align(pred, target, mask)
    m = mask.bool()
    loss = F.smooth_l1_loss(aligned[m], target[m])
    rmsd = torch.sqrt((((aligned - target) ** 2).sum(-1) * mask).sum()
                      / mask.sum().clamp_min(1.0))
    return loss, rmsd


# --------------------------------------------------------------------------- #
def load_example(fid: str) -> Dict[str, torch.Tensor]:
    with open(TEST_DIR / "inputs.tsv") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    r = rows[fid]
    pdb = TEST_DIR / "pdbs" / "alphafold" / fid / f"{fid}_model_1_model_2_ptm.pdb"
    return parse.parse_example(pdb, r["peptide"], r["mhc_seq"],
                               r["anchors"], r["mhc_type"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3GSO_0_0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--no-lr-decay", action="store_true",
                    help="disable cosine LR decay (decay stabilises the tail)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    ex = load_example(args.id)
    aatype = ex["aatype"].to(device)
    segment_id = ex["segment_id"].to(device)
    anchor = ex["anchor"].to(device)
    residue_index = ex["residue_index"].to(device)
    seq_mask = ex["seq_mask"].to(device)
    teacher_ca = ex["teacher_ca"].to(device)
    n = int(aatype.shape[0])
    print(f"[overfit] id={args.id}  N={n}  n_mhc={ex['n_mhc']}  n_pep={ex['n_pep']}"
          f"  device={device}")

    encoder = ThrowawayEncoder().to(device)
    frozen = load_frozen_fold(device=str(device))
    assert not any(p.requires_grad for p in frozen.parameters()), \
        "frozen stack must have no trainable parameters"
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    sched = (None if args.no_lr_decay else
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps))

    curve: list[tuple[int, float, float]] = []
    for step in range(args.steps + 1):
        s, z = encoder(aatype, segment_id, anchor, residue_index)
        ca, _, _ = frozen(s[None], z[None], aatype[None], seq_mask[None])
        loss, rmsd = superposed_loss(ca[0], teacher_ca, seq_mask)
        if step % args.print_every == 0 or step == args.steps:
            curve.append((step, float(loss), float(rmsd)))
            print(f"  step {step:4d}  loss {float(loss):.4f}  "
                  f"Cα-RMSD {float(rmsd):.3f} Å")
        if step == args.steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

    r0, rf = curve[0][2], curve[-1][2]
    print(f"\n[overfit] RMSD {r0:.3f} -> {rf:.3f} Å over {args.steps} steps "
          f"({'PASS <2 Å' if rf < 2.0 else 'HIGH PLATEAU — investigate'})")


if __name__ == "__main__":
    main()
