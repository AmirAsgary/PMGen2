"""Inference latency for model_multimer_1, with a per-stage breakdown.

The design target is 10-50 ms per structure. Nothing in this repo has ever measured
it: predict_test.py reports RMSDs only, and the training log's 2.7 it/s is a
forward+backward at bs=1 with data loading, which says little about inference.

This times `MultimerModel.predict` (the deployment path: atom14 + pLDDT logits, no
grad) and then attributes the time to embedder / head-2 / trunk / frozen SM / pLDDT,
so an over-target result points at what to change rather than just failing.

  $PY src/model_multimer_1/bench_latency.py                     # defaults
  $PY src/model_multimer_1/bench_latency.py --ckpt <last.pt> --batch 1 4 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import model as MM                                                    # noqa: E402


def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, n_warm: int, n_iter: int, dev: str) -> float:
    """Median ms per call. Median, not mean: one stray context switch on a shared
    node otherwise dominates a 10 ms measurement."""
    for _ in range(n_warm):
        fn()
    _sync(dev)
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="optional trainable-state checkpoint")
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--n-mhc", type=int, default=180)
    p.add_argument("--n-pep", type=int, default=9)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--pep-frames", choices=["teacher", "identity"], default="identity")
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (args.dtype == "bf16" and dev.startswith("cuda"))
    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev,
                           pep_frames=args.pep_frames)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("trainable", ck.get("model", ck))
        bad = [k for k, v in sd.items()
               if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            raise SystemExit(f"FATAL: checkpoint has {len(bad)}/{len(sd)} non-finite "
                             f"tensors (e.g. {bad[:3]}) — it is dead, not slow.")
        net.load_state_dict(sd, strict=False)
    net.eval()

    print(f"device={dev} dtype={args.dtype} n_trunk={args.n_trunk} "
          f"N={args.n_mhc + args.n_pep} ({args.n_mhc} MHC + {args.n_pep} peptide)")
    if dev == "cpu":
        print("WARNING: no GPU here — CPU numbers say nothing about the 10-50 ms target.")
    print(f"{'batch':>6} {'ms/batch':>10} {'ms/structure':>13} {'target 10-50ms':>15}")

    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp \
        else torch.no_grad
    for B in args.batch:
        batch = MM._dummy_batch(dev, B=B, n_mhc=args.n_mhc, n_pep=args.n_pep)

        def run():
            with torch.no_grad():
                if amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        net.predict(batch)
                else:
                    net.predict(batch)

        ms = _time(run, args.warmup, args.iters, dev)
        per = ms / B
        verdict = "OK" if per <= 50 else ("over" if per <= 100 else "WAY over")
        print(f"{B:>6} {ms:>10.1f} {per:>13.1f} {verdict:>15}")

    # ---- per-stage breakdown at batch 1 (where the target is defined) ----
    print("\nper-stage breakdown @ batch 1 (median ms):")
    batch = MM._dummy_batch(dev, B=1, n_mhc=args.n_mhc, n_pep=args.n_pep)

    def stage(fn):
        def run():
            with torch.no_grad():
                if amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fn()
                else:
                    fn()
        return _time(run, args.warmup, args.iters, dev)

    f = MM.build_multimer_feats(batch["aatype"], batch["segment_id"], batch["seq_mask"])
    idm = (MM.m1.peptide_mask_from_batch(batch["seq_mask"], batch["segment_id"])
           if args.pep_frames == "identity" else None)
    frames = MM._frames_from_bb(batch["teacher_bb"], batch["seq_mask"],
                                noise=0.0, identity_mask=idm)
    parts = [
        ("build_multimer_feats", lambda: MM.build_multimer_feats(
            batch["aatype"], batch["segment_id"], batch["seq_mask"])),
        ("embedder (head-1)", lambda: net.embedder(f)),
        ("head-2 (MHC enc)", lambda: net.head2(batch["teacher_bb"], f["mhc"], frames)),
        ("_encode (all of the above + trunk)", lambda: net._encode(batch)),
        ("FULL predict()", lambda: net.predict(batch)),
    ]
    res = {}
    for name, fn in parts:
        res[name] = stage(fn)
        print(f"  {name:<38} {res[name]:7.1f}")
    enc = res["_encode (all of the above + trunk)"]
    full = res["FULL predict()"]
    print(f"  {'-> frozen SM + torsions + pLDDT (rest)':<38} {full - enc:7.1f}")
    print(f"\n  encoder path : {enc / full * 100:5.1f}% of latency (trainable)")
    print(f"  frozen AF end: {(full - enc) / full * 100:5.1f}% of latency "
          f"(FROZEN — not reducible by training, only by changing the decoder)")


if __name__ == "__main__":
    main()
