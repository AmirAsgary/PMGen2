"""Is bs>1 CORRECT, and how much faster is it?

This project has only ever trained at bs=1, so the batch dimension is effectively
untested. That is not hypothetical: openfold's supervised_chi_loss computes
chi_pi_periodic with einsum("...ij,jk->ik"), which DROPS the leading batch dim and
silently sums over it — invisible at bs=1, wrong above it. (Our per-example version
fixes that; other paths may not be so lucky.)

CORRECTNESS: run a batch of K structures together, then the SAME K one at a time, in
eval mode (no dropout, no MHC noise) so both are deterministic. A correct implementation
gives batched_loss == mean(individual_losses). Any mismatch is a batch-dim bug.

SPEED: examples/second and peak memory per batch size — the number that decides whether
a bigger batch is worth changing a training configuration that currently works.

    $PY src/model_multimer_1/bench_batch.py --sizes 1 2 4 8
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd, torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "openfold")); sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_HERE))
import utils as m1                                                    # noqa: E402
import model as MM                                                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--n-trunk", type=int, default=3)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    kw = {}
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location=dev, weights_only=False)["trainable"]
        kw = dict(attn_norm=any("attn_norm" in k for k in sd),
                  plddt_adapter=any("plddt_proj" in k for k in sd))
    net = MM.MultimerModel(n_trunk=a.n_trunk, device=dev, pep_frames="identity", **kw)
    net.set_stage(1)
    if a.ckpt:
        net.load_state_dict(sd, strict=False)
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                              lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)

    idx = pd.read_csv(Path(a.h5_dir) / "index.csv", dtype=str).head(200)
    ds = m1.H5DistillDataset(idx["id"].tolist(), dict(zip(idx["id"], idx["shard"])), a.h5_dir)

    def timed_step_loop(bs):
        """One fwd+bwd per iteration, graph FREED each time.

        The obvious version — collect the losses, then backward them in a second loop —
        keeps `iters` autograd graphs alive at once and inflates peak memory by that
        factor. It made bs=4 OOM here while the real training job sat at 4.3 GB.
        """
        net.train(True)
        ld = m1.make_dataloader(ds, bs, shuffle=False, num_workers=2)
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        n = 0
        t0 = time.perf_counter()
        for i, b in enumerate(ld):
            if i >= a.iters: break
            b = m1.move_batch(b, dev)
            net.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = net(b, return_frames=True)
                tot, _ = loss_mod(o[0], o[1], o[2], o[3], b, aux=o[4])
            tot.backward()                 # graph released here
            del o, tot
            n += 1
        if dev == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / max(n, 1)
        gb = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        return dt, gb

    # ---------- CORRECTNESS ----------
    print("=== CORRECTNESS: batched loss vs mean of individual losses (eval mode) ===")
    print(f"{'bs':>3} {'batched':>10} {'mean(indiv)':>12} {'abs diff':>10}  verdict")
    net.eval()
    with torch.no_grad():
        for bs in [s for s in a.sizes if s > 1]:
            ld = m1.make_dataloader(ds, bs, shuffle=False, num_workers=2)
            batch = m1.move_batch(next(iter(ld)), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = net(batch, return_frames=True)
                batched, _ = loss_mod(o[0], o[1], o[2], o[3], batch, aux=o[4])
            singles = []
            ld1 = m1.make_dataloader(ds, 1, shuffle=False, num_workers=2)
            for i, b1 in enumerate(ld1):
                if i >= bs: break
                b1 = m1.move_batch(b1, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o1 = net(b1, return_frames=True)
                    t1, _ = loss_mod(o1[0], o1[1], o1[2], o1[3], b1, aux=o1[4])
                singles.append(float(t1))
            mi = sum(singles) / len(singles)
            d = abs(float(batched) - mi)
            ok = "OK" if d < 0.02 else ("SUSPECT" if d < 0.1 else "*** MISMATCH ***")
            print(f"{bs:>3} {float(batched):>10.4f} {mi:>12.4f} {d:>10.4f}  {ok}")
    print("  (small diffs are expected: padding to the batch max changes masked means")
    print("   slightly, and bf16 reassociates. A LARGE diff means a batch-dim bug.)")

    # ---------- SPEED ----------
    print(f"\n=== SPEED (fwd+bwd, train mode) ===")
    print(f"{'bs':>3} {'s/step':>9} {'examples/s':>12} {'peak GB':>9} {'vs bs=1':>9}")
    base = None
    for bs in a.sizes:
        try:
            step, gb = timed_step_loop(bs)
            eps = bs / step
            base = base or eps
            print(f"{bs:>3} {step:>9.3f} {eps:>12.2f} {gb:>9.2f} {eps/base:>8.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>3} {'OOM':>9}  (真 peak, one graph at a time)".replace("真 ",""))
            torch.cuda.empty_cache(); break


if __name__ == "__main__":
    main()
