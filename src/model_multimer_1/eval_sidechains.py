"""
Acceptance test for the side-chain objective.

Measures, on a HELD-OUT set, the predicted torsions from the trainable angle head against
the teacher's chi. Reports each chi's "% within 40 deg" NEXT TO the uninformative baseline
— a number without its null is meaningless (chi1 was 10.8% against a 22% random baseline
before this work, i.e. worse than guessing).

chi1 is never 180-periodic, so it is the clean read. chi2..4 are compared symmetry-aware
(chi_pi_periodic), which halves their effective random baseline to ~44%.

  $PY src/model_multimer_1/eval_sidechains.py --ckpt <last.pt> --hasmig-dir <store> \
      --max-train 60 --subset-val-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "openfold"))
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_HERE))

import utils as m1                                                   # noqa: E402
import model as MM                                                   # noqa: E402
from openfold.np import residue_constants as rc                      # noqa: E402

RANDOM_BASELINE = {1: 22.2, 2: 44.4, 3: 44.4, 4: 44.4}   # % within 40 deg, uninformative


def held_out_ids(store: Path, max_train: int, val_frac: float, seed: int):
    """Reproduce train.py's disjoint subset split exactly (same seed / same order)."""
    idx = pd.read_csv(store / "index.csv", dtype=str)
    ids = idx["id"].tolist()
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    order = order[: min(max_train, len(order))]
    n_val = max(1, int(round(len(order) * val_frac)))
    val = sorted(order[:n_val])
    return [ids[i] for i in val], dict(zip(idx["id"], idx["shard"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="omit to score the UNTRAINED model")
    p.add_argument("--hasmig-dir", required=True)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--subset-val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--stage", type=int, default=1)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vids, id2shard = held_out_ids(Path(args.hasmig_dir), args.max_train,
                                  args.subset_val_frac, args.seed)
    ds = m1.H5DistillDataset(vids, id2shard, args.hasmig_dir)

    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames="identity")
    net.set_stage(args.stage)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=dev,
                                       weights_only=False)["trainable"], strict=False)
    net.eval()

    periodic = torch.tensor(rc.chi_pi_periodic, dtype=torch.float32, device=dev)
    errs = {1: [], 2: [], 3: [], 4: []}
    # MHC vs PEPTIDE must be reported separately. The MHC is ~95% of the residues and is
    # often the SAME protein across a store, so a pooled chi number can be near-perfect
    # purely by memorising one allele's rotamers. Only the peptide chi is a real read.
    errs_mhc = {1: [], 2: [], 3: [], 4: []}
    errs_pep = {1: [], 2: [], 3: [], 4: []}
    pep_rmsd = []
    for i in range(len(ds)):
        b = m1.move_batch(m1.collate_with_teacher([ds[i]]), dev)
        with torch.no_grad():
            _, _, _, _, aux = net(b, return_frames=True)
        pred = aux["angles"][:, :, 3:, :]                 # chi1..4 (sin,cos)
        true = b["teacher_chi"]
        mask = b["teacher_chi_mask"] * b["seq_mask"][..., None]
        ang = lambda t: torch.atan2(t[..., 0], t[..., 1])
        d = (ang(pred) - ang(true)).abs()
        d = torch.minimum(d, 2 * np.pi - d)
        per = periodic[b["aatype"].long()].bool()          # [B,N,4]
        d = torch.where(per, torch.minimum(d, np.pi - d), d)
        d = d * 180.0 / np.pi
        is_pep = m1.peptide_mask_from_batch(b["seq_mask"], b["segment_id"]).bool()
        for c in (1, 2, 3, 4):
            m = mask[..., c - 1] > 0.5
            if m.any():
                errs[c].append(d[..., c - 1][m].cpu().numpy())
            mm = m & ~is_pep
            mp = m & is_pep
            if mm.any():
                errs_mhc[c].append(d[..., c - 1][mm].cpu().numpy())
            if mp.any():
                errs_pep[c].append(d[..., c - 1][mp].cpu().numpy())
        # peptide pose (binding-pose convention: superpose on MHC)
        pep = m1.peptide_mask_from_batch(b["seq_mask"], b["segment_id"]).bool()[0]
        mhc = b["seq_mask"].bool()[0] & ~pep
        r = m1._superpose_rmsd_on(aux["atom14"][0, :, 1, :].float(),
                                  b["teacher_ca"][0].float(), mhc, pep)
        if r is not None:
            pep_rmsd.append(r.item())

    tag = Path(args.ckpt).parent.name if args.ckpt else "UNTRAINED"
    n_mhc_seq = len({tuple(ds[i]["aatype"][: ds[i]["n_mhc"]].tolist()) for i in range(len(ds))})
    print(f"\n=== side-chain acceptance test: {tag} | {len(ds)} HELD-OUT structures ===")
    if n_mhc_seq < max(2, len(ds) // 4):
        print(f"!! WARNING: only {n_mhc_seq} distinct MHC sequence(s) in the held-out set.")
        print("!! The MHC is ~95% of residues, so the POOLED chi number can be near-perfect")
        print("!! by memorising one allele's rotamers. Read the PEPTIDE column.")

    def table(name, e, crit):
        print(f"\n-- {name} --")
        print(f"{'angle':<7} {'median err':>11} {'% within 40°':>13} {'random':>9}  verdict")
        first = None
        for c in (1, 2, 3, 4):
            if not e[c]:
                continue
            a = np.concatenate(e[c])
            pct = 100.0 * (a < 40).mean()
            base = RANDOM_BASELINE[c]
            if c == 1:
                first = pct > base
            print(f"chi{c:<4} {np.median(a):>10.1f}° {pct:>12.1f}% {base:>8.1f}%  "
                  f"{'PASS' if pct > base else 'below baseline'}  (n={len(a)})")
        return first

    table("POOLED (MHC + peptide) — inflated when the MHC repeats", errs, False)
    table("MHC only", errs_mhc, False)
    pep_ok = table("PEPTIDE only  <-- the real generalisation read", errs_pep, True)

    print(f"\npeptide Cα-RMSD (held-out, superposed on MHC): {np.mean(pep_rmsd):.2f} Å")
    print(f"\nACCEPTANCE (PEPTIDE chi1 > {RANDOM_BASELINE[1]}% random): "
          f"{'PASS' if pep_ok else 'FAIL — not learning peptide side chains'}")
    return 0 if pep_ok else 1


if __name__ == "__main__":
    sys.exit(main())
