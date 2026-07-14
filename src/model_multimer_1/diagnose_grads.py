"""
Short gradient diagnostic for the mm1 training instability.

The 13_08 run trained well for 24k steps, then ran away exponentially over ~300 steps
and died. `grad_clip=1.0` was on, yet the gradients we now log are 30-400x that. This
script answers WHERE those gradients come from and WHETHER there is a feedback loop.

It records, per step (bs=1, so one structure per step):
  * total gradient norm, and the norm PER MODULE (embedder / head2 / trunk / sm_s / sm_z /
    angle_head / plddt_proj) -> which part of the model explodes?
  * ||unnormalized angles|| from the AngleResnet. The head predicts torsions as
    s / ||s||; the gradient of that scales like 1/||s||, so if ||s|| drifts toward 0 the
    gradient blows up -> a genuine runaway mechanism. AF keeps ||s|| near 1 with the
    `angle_norm` penalty (weight 0.01). Is ours holding?
  * the loss terms and the structure id -> are a few pathological structures responsible?

  $PY src/model_multimer_1/diagnose_grads.py --hasmig-dir <store> --steps 600
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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

MODULES = ["embedder", "head2", "s_proj", "z_proj", "trunk",
           "sm_s", "sm_z", "angle_head", "plddt_proj"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hasmig-dir", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--bs", type=int, default=1)          # match the failed run
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--resume", default=None, help="optional trained ckpt")
    p.add_argument("--out", default="outputs/mm1_graddiag")
    p.add_argument("--dtype", choices=["bf16","fp32"], default="bf16")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = pd.read_csv(Path(args.hasmig_dir) / "index.csv", dtype=str)
    ds = m1.H5DistillDataset(idx["id"].tolist(), dict(zip(idx["id"], idx["shard"])),
                             args.hasmig_dir)
    loader = m1.make_dataloader(ds, args.bs, shuffle=True, num_workers=2)

    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames="identity")
    net.set_stage(1)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=dev,
                                       weights_only=False)["trainable"], strict=False)
    net.train()
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                              lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)
    opt = torch.optim.AdamW(net.trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    rows = []
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            ids = batch.get("id", ["?"])
            batch = m1.move_batch(batch, dev)
            opt.zero_grad(set_to_none=True)
            use_amp = args.dtype == "bf16"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                ca, pl, pae, fr, aux = net(batch, return_frames=True)
                total, terms = loss_mod(ca, pl, pae, fr, batch, aux=aux)
            total.backward()

            # ---- gradient norm, total and PER MODULE (before any clipping) ----
            rec = {"step": step, "id": ids[0], "total": float(total)}
            for k in ("fape", "sc_fape", "chi_loss"):
                rec[k] = float(terms[k])
            g2 = 0.0
            for mname in MODULES:
                mod = getattr(net, mname)
                s = sum(float((p.grad.detach() ** 2).sum())
                        for p in mod.parameters() if p.grad is not None)
                rec[f"g_{mname}"] = s ** 0.5
                g2 += s
            rec["g_total"] = g2 ** 0.5

            # ---- the AngleResnet normalisation: ||unnormalized angles|| ----
            un = aux["unnormalized_angles"].detach().float()          # [B,N,7,2]
            nrm = un.norm(dim=-1)                                     # [B,N,7]
            m = batch["seq_mask"].bool()[..., None].expand_as(nrm)
            v = nrm[m]
            rec["unnorm_min"] = float(v.min())
            rec["unnorm_mean"] = float(v.mean())
            rec["n_res"] = int(batch["seq_mask"].sum())
            rec["n_pep"] = int(m1.peptide_mask_from_batch(
                batch["seq_mask"], batch["segment_id"]).sum())
            rows.append(rec)

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.trainable_parameters(), args.grad_clip)
            opt.step()
            step += 1

    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "grad_diag.csv", index=False)

    g = df.g_total
    print(f"\n================ {len(df)} steps (bs={args.bs}, lr={args.lr}) ================")
    print(f"TOTAL grad norm   median {g.median():8.2f} | p90 {g.quantile(.9):8.2f} "
          f"| p99 {g.quantile(.99):8.2f} | MAX {g.max():9.2f}")
    print(f"  (grad_clip = {args.grad_clip}: {100*(g>args.grad_clip).mean():.0f}% of steps "
          f"are clipped; max is {g.max()/args.grad_clip:.0f}x the clip)")

    print(f"\n--- WHICH MODULE produces the gradient? (median share of ||g||^2) ---")
    tot2 = (df[[f'g_{m}' for m in MODULES]] ** 2).sum(axis=1)
    for mname in MODULES:
        share = ((df[f"g_{mname}"] ** 2) / tot2.clip(lower=1e-12)).median() * 100
        print(f"  {mname:12} {share:5.1f}%   median |g| {df[f'g_{mname}'].median():8.3f} "
              f"  max {df[f'g_{mname}'].max():10.2f}")

    print(f"\n--- AngleResnet normalisation: ||unnormalized angles|| ---")
    print(f"  mean over steps: {df.unnorm_mean.mean():.4f}   (AF keeps this near 1.0)")
    print(f"  MIN seen       : {df.unnorm_min.min():.2e}   <- the 1/||s|| gradient blows "
          f"up as this -> 0")
    c = np.corrcoef(df.unnorm_min, df.g_total)[0, 1]
    print(f"  corr(min ||s||, total grad norm) = {c:+.3f}")
    lo = df[df.unnorm_min < df.unnorm_min.quantile(.1)]
    hi = df[df.unnorm_min > df.unnorm_min.quantile(.9)]
    print(f"  steps with the SMALLEST ||s|| -> median grad {lo.g_total.median():8.2f}")
    print(f"  steps with the LARGEST  ||s|| -> median grad {hi.g_total.median():8.2f}")

    print(f"\n--- are a few STRUCTURES responsible? (top-5 grad spikes) ---")
    top = df.nlargest(5, "g_total")
    for _, r in top.iterrows():
        print(f"  step {int(r.step):>4} |g|={r.g_total:9.2f}  fape={r.fape:6.3f} "
              f"chi={r.chi_loss:6.3f} n_pep={int(r.n_pep)} min||s||={r.unnorm_min:.2e} "
              f"id={r.id}")
    rep = df.id.value_counts()
    worst = df.groupby("id").g_total.max().nlargest(3)
    print(f"\n  structures seen {rep.iloc[0]}x on average; worst-3 by max |g|:")
    for i, v in worst.items():
        print(f"    {v:9.2f}  {i}")
    print(f"\nwrote {out/'grad_diag.csv'}")


if __name__ == "__main__":
    main()
