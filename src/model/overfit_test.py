"""
GATE 3 — overfit one --dummy example with the FULL 3-term loss (variant 7).

Optimizes the encoder only; the frozen SM + heads stay frozen. Expect FAPE ↓,
Cα-RMSD < ~2 Å, pLDDT Spearman → ~1, PAE MAE ↓. Prints the curves.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/model/overfit_test.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3GSO_0_0")
    ap.add_argument("--variant", type=int, default=7)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--print-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", type=float, nargs=3, default=(1.0, 0.1, 0.1),
                    metavar=("FAPE", "PLDDT", "PAE"))
    args = ap.parse_args()
    U.set_seed(args.seed)

    # build the single requested example via the dummy data path
    row = next(r for r in U.dummy_rows() if r["id"] == args.id)
    item = U.DistillDataset([row])[0]
    batch = U.collate_with_teacher([item])
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    n = int(batch["aatype"].shape[1])
    print(f"[overfit] id={args.id} variant={args.variant} N={n} device={DEVICE} "
          f"lambdas(fape,plddt,pae)={tuple(args.lambdas)}")

    model = U.DistillModel(args.variant, device=DEVICE).to(DEVICE)
    model.train()
    loss_mod = U.DistillLoss(*args.lambdas).to(DEVICE)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    first = last = None
    print(f"  {'step':>5} {'total':>7} {'fape':>7} {'plddt_ce':>8} "
          f"{'pae_ce':>7} | {'CaRMSD':>7} {'pLDDT_sp':>8} {'PAE_MAE':>7}")
    for step in range(args.steps + 1):
        ca, plddt, pae, frames = model(batch, return_frames=True)
        total, terms = loss_mod(ca, plddt, pae, frames, batch)
        if step % args.print_every == 0 or step == args.steps:
            mets = U.eval_metrics(ca, plddt, pae, batch, loss_mod)
            print(f"  {step:5d} {float(terms['total']):7.3f} "
                  f"{float(terms['fape']):7.3f} {float(terms['plddt_ce']):8.3f} "
                  f"{float(terms['pae_ce']):7.3f} | {mets['ca_rmsd']:7.3f} "
                  f"{mets['plddt_spearman']:8.3f} {mets['pae_mae']:7.3f}")
            rec = (float(terms["fape"]), mets["ca_rmsd"],
                   mets["plddt_spearman"], mets["pae_mae"])
            first = first or rec
            last = rec
        if step == args.steps:
            break
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

    f0, r0, s0, m0 = first
    f1, r1, s1, m1 = last
    print(f"\n[overfit] FAPE {f0:.3f}->{f1:.3f} | Cα-RMSD {r0:.2f}->{r1:.2f} Å | "
          f"pLDDT Spearman {s0:.2f}->{s1:.2f} | PAE MAE {m0:.2f}->{m1:.2f} Å")
    ok = (f1 < f0 and r1 < 2.0 and s1 > 0.9 and m1 < m0)
    print("GATE 3 " + ("PASSED" if ok else "NEEDS REVIEW")
          + ": FAPE↓, Cα-RMSD<2Å, pLDDT Spearman→~1, PAE MAE↓.")


if __name__ == "__main__":
    main()
