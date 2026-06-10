"""
GATE 1 — encoder verification (PART 1).

For every variant 1..7: forward on a real --dummy example and on a padded batch
of 2; assert encoder outputs s[B,N,384] / z[B,N,N,128] and a full frozen pass ->
ca[B,N,3]; assert gradient reaches the encoder and NO frozen param has grad;
assert padding does not leak (perturbing padded positions leaves real outputs
unchanged). Prints a per-variant table: #encoder params, fwd ms, peak mem.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/model/encoder_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SINGLE_ID = "3GSO_0_0"          # N=194
PAIR_IDS = ["3GSO_0_0", "6UK4_0_0"]   # N=194 and N=193 -> 1 padded position


def _to_device(batch: dict) -> dict:
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _make_batches():
    single = U.collate_fn([U.load_dummy_examples([SINGLE_ID])[0]])
    pair = U.collate_fn(U.load_dummy_examples(PAIR_IDS))
    return _to_device(single), _to_device(pair)


def check_shapes(model: U.DistillModel, batch: dict, tag: str) -> None:
    b, n = batch["aatype"].shape
    s, z = model.encoder(batch["aatype"], batch["residue_index"],
                         batch["seq_mask"], batch["anchor"], batch["segment_id"])
    assert s.shape == (b, n, U.FROZEN_C_S), f"{tag}: s {tuple(s.shape)}"
    assert z.shape == (b, n, n, U.FROZEN_C_Z), f"{tag}: z {tuple(z.shape)}"
    ca, plddt, pae = model(batch)
    assert ca.shape == (b, n, 3), f"{tag}: ca {tuple(ca.shape)}"
    assert plddt.shape == (b, n, 50), f"{tag}: plddt {tuple(plddt.shape)}"
    assert pae.shape == (b, n, n, 64), f"{tag}: pae {tuple(pae.shape)}"


def check_grad_isolation(model: U.DistillModel, batch: dict) -> None:
    model.zero_grad(set_to_none=True)
    ca, _, _ = model(batch)
    loss = (ca * batch["seq_mask"][..., None]).sum()
    loss.backward()
    enc_with_grad = sum(p.grad is not None and torch.any(p.grad != 0)
                        for p in model.encoder.parameters())
    assert enc_with_grad > 0, "no encoder parameter received gradient"
    for p in model.frozen.parameters():
        assert not p.requires_grad, "frozen param has requires_grad=True"
        assert p.grad is None, "frozen param received a gradient"
    model.zero_grad(set_to_none=True)


def check_no_padding_leak(model: U.DistillModel, pair_batch: dict) -> float:
    """Perturb the padded slots of the shorter example; assert every real output
    (encoder s/z and frozen ca) is unchanged. Returns the max abs deviation."""
    lengths = pair_batch["length"].tolist()
    short = int(torch.tensor(lengths).argmin())     # example with padding
    n_real = lengths[short]
    max_n = pair_batch["aatype"].shape[1]
    assert n_real < max_n, "test batch has no padded positions"
    pad = slice(n_real, max_n)

    with torch.no_grad():
        s0, z0 = model.encoder(pair_batch["aatype"], pair_batch["residue_index"],
                               pair_batch["seq_mask"], pair_batch["anchor"],
                               pair_batch["segment_id"])
        ca0, _, _ = model(pair_batch)

        pert = {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in pair_batch.items()}
        g = torch.Generator(device=DEVICE).manual_seed(123)
        pert["aatype"][short, pad] = torch.randint(0, U.N_AATYPE, (max_n - n_real,),
                                                   device=DEVICE, generator=g)
        pert["anchor"][short, pad] = torch.randint(0, 2, (max_n - n_real,),
                                                   device=DEVICE, generator=g).float()
        pert["segment_id"][short, pad] = torch.randint(0, U.N_SEGMENTS,
                                                       (max_n - n_real,),
                                                       device=DEVICE, generator=g)
        pert["residue_index"][short, pad] = torch.randint(
            0, 500, (max_n - n_real,), device=DEVICE, generator=g)
        s1, z1 = model.encoder(pert["aatype"], pert["residue_index"],
                               pert["seq_mask"], pert["anchor"], pert["segment_id"])
        ca1, _, _ = model(pert)

    # real region of the perturbed example + the entirely-real other example
    devs = []
    devs.append((s1[short, :n_real] - s0[short, :n_real]).abs().max())
    devs.append((z1[short, :n_real, :n_real] - z0[short, :n_real, :n_real]).abs().max())
    devs.append((ca1[short, :n_real] - ca0[short, :n_real]).abs().max())
    other = 1 - short
    devs.append((s1[other] - s0[other]).abs().max())
    devs.append((ca1[other] - ca0[other]).abs().max())
    max_dev = float(torch.stack(devs).max())
    assert max_dev < 1e-4, f"padding leaked: max real-output deviation {max_dev:.2e}"
    return max_dev


def time_forward(model: U.DistillModel, batch: dict, iters: int = 10) -> tuple[float, float]:
    for _ in range(3):                      # warmup
        model(batch)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(batch)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    mem = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0
    return ms, mem


def main() -> None:
    U.set_seed(0)
    single, pair = _make_batches()
    print(f"device={DEVICE}  single N={single['aatype'].shape[1]}  "
          f"pair shape={tuple(pair['aatype'].shape)} "
          f"(lengths {pair['length'].tolist()})\n")

    rows = []
    for variant in range(1, 8):
        U.set_seed(0)
        model = U.DistillModel(variant, device=DEVICE).to(DEVICE)
        model.train()
        check_shapes(model, single, f"variant{variant}/single")
        check_shapes(model, pair, f"variant{variant}/pair")
        check_grad_isolation(model, pair)
        max_dev = check_no_padding_leak(model, pair)
        ms, mem = time_forward(model, pair)
        n_params = sum(p.numel() for p in model.encoder.parameters())
        rows.append((variant, n_params, ms, mem, max_dev))
        print(f"  variant {variant}: OK  "
              f"(params {n_params/1e6:.2f}M, fwd {ms:.1f} ms, "
              f"peak {mem:.0f} MB, leak {max_dev:.1e})")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'variant':8} {'enc_params':12} {'fwd_ms(B=2)':12} "
          f"{'peak_MB':10} {'leak_dev':10}")
    for v, p, ms, mem, dev in rows:
        print(f"{v:<8} {p:<12,} {ms:<12.1f} {mem:<10.0f} {dev:<10.1e}")
    print("\nALL 7 VARIANTS PASSED (shapes, grad isolation, no padding leak).")


if __name__ == "__main__":
    main()
