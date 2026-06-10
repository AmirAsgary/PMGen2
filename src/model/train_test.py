"""
GATE 4 — short --dummy training run over all 15 examples.

Asserts (1) the loss correctly ignores padded teacher arrays (perturbing padded
positions of teacher pLDDT/PAE/backbone leaves the loss unchanged), and (2) the
training loss decreases over a few epochs (batch 3). Prints a per-epoch per-term
table.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/model/train_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_padded_teacher_masking() -> None:
    """Perturb ONLY the padded slots of the teacher arrays; the masked 3-term
    loss must be unchanged (real model inputs untouched)."""
    U.set_seed(0)
    ds = U.build_dataset(dummy=True)
    items = [ds[i] for i in range(len(ds))]
    by_len = {int(e["aatype"].shape[0]): e for e in items}
    a, b = by_len[max(by_len)], by_len[min(by_len)]          # 194 and 193
    batch = U.move_batch(U.collate_with_teacher([a, b]), DEVICE)
    n_b = int(b["aatype"].shape[0])
    max_n = batch["aatype"].shape[1]
    assert n_b < max_n, "need a padded example"

    model = U.DistillModel(7, device=DEVICE).to(DEVICE)
    loss_mod = U.DistillLoss().to(DEVICE)
    with torch.no_grad():
        ca, pl, pae, fr = model(batch, return_frames=True)
        base = float(loss_mod(ca, pl, pae, fr, batch)[0])

        pert = {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in batch.items()}
        g = torch.Generator(device=DEVICE).manual_seed(1)
        pad = slice(n_b, max_n)
        pert["teacher_plddt"][1, pad] = torch.rand(max_n - n_b, device=DEVICE,
                                                   generator=g) * 100
        pert["teacher_pae"][1, pad, :] = torch.rand(max_n - n_b, max_n,
                                                    device=DEVICE, generator=g) * 30
        pert["teacher_pae"][1, :, pad] = torch.rand(max_n, max_n - n_b,
                                                    device=DEVICE, generator=g) * 30
        pert["teacher_bb"][1, pad] = torch.randn(max_n - n_b, 3, 3, device=DEVICE,
                                                 generator=g) * 10
        # same model outputs (inputs unchanged) -> loss must match
        loss_pert = float(loss_mod(ca, pl, pae, fr, pert)[0])

    dev = abs(loss_pert - base)
    print(f"=== padded-teacher masking ===  loss {base:.6f} vs perturbed "
          f"{loss_pert:.6f}  (|Δ|={dev:.2e})")
    assert dev < 1e-4, f"loss leaked from padded teacher positions: Δ={dev:.2e}"
    print("  OK: padded teacher pLDDT/PAE/backbone do not affect the loss.\n")


def check_training_decreases() -> None:
    print("=== short --dummy training (variant 7, bs=3, 12 epochs) ===")
    history, _ = U.run_training(variant=7, dummy=True, epochs=12, bs=3, lr=3e-3,
                                seed=0, device=DEVICE, grad_clip=1.0)
    print(f"\n  {'epoch':>5} {'tr_total':>8} {'tr_fape':>8} {'tr_plddt':>8} "
          f"{'tr_pae':>7} | {'val_RMSD':>8} {'val_sp':>7} {'val_MAE':>7}")
    for h in history:
        t, v = h["train"], h["val"]
        print(f"  {h['epoch']:5d} {t['total']:8.3f} {t['fape']:8.3f} "
              f"{t['plddt_ce']:8.3f} {t['pae_ce']:7.3f} | {v['ca_rmsd']:8.2f} "
              f"{v['plddt_spearman']:7.2f} {v['pae_mae']:7.2f}")
    first, last = history[0]["train"]["total"], history[-1]["train"]["total"]
    print(f"\n  train total {first:.3f} -> {last:.3f}")
    assert last < first, f"training loss did not decrease ({first:.3f}->{last:.3f})"
    print("  OK: training loss decreased.\n")


def main() -> None:
    check_padded_teacher_masking()
    check_training_decreases()
    print("GATE 4 PASSED: padded-teacher masking correct + loss decreases.")


if __name__ == "__main__":
    main()
