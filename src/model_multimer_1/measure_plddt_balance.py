"""What should lambda_plddt be, now that the pLDDT CE reaches the ENCODER?

Stage 2 inherited lambda_plddt = 0.01 from a design where a trainable adapter sat in
front of the frozen pLDDT head. That adapter is gone (AlphaFold feeds the head the SM's
`single` directly), so the CE gradient no longer terminates in a 384x384 projection —
it flows through the frozen head, back through the frozen StructureModule, and into the
trunk. Same lambda, different path, so the balance has to be re-measured rather than
inherited.

Measures on REAL batches, separately:
    ||g_struct||  from  bb-FAPE + sc-FAPE + chi        (what stage 1 optimised)
    ||g_plddt||   from  the pLDDT cross-entropy alone
over the ENCODER parameters only (the frozen SM/head have no gradients).

Reports the lambda that puts the confidence gradient at a chosen FRACTION of the
structural one — i.e. strong enough to learn confidence, weak enough not to wreck the
structure the run just spent hours getting right.

    $PY src/model_multimer_1/measure_plddt_balance.py --ckpt <ckpt> --n 24
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "openfold")); sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_HERE))
import utils as m1                                                    # noqa: E402
import model as MM                                                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm")
    p.add_argument("--plddt-peptide-ratio", type=float, default=0.0,
                   help="peptide:MHC ratio inside the pLDDT CE (0 = use peptide_weight). "
                        "Changing it changes the CE MAGNITUDE (the masked mean is "
                        "normalised by the weight sum), so the lambda balance must be "
                        "re-measured rather than carried over.")
    p.add_argument("--target-frac", type=float, default=0.10,
                   help="desired ||lambda*g_plddt|| / ||g_struct||")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)["trainable"]
    net = MM.MultimerModel(n_trunk=a.n_trunk, device=dev, pep_frames="identity",
                           angle_input=a.angle_input,
                           attn_norm=any("attn_norm" in k for k in sd),
                           plddt_adapter=any("plddt_proj" in k for k in sd))
    net.set_stage(2)
    net.load_state_dict(sd, strict=False)
    net.train()

    struct = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                            lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)
    conf = m1.DistillLoss(0.0, 1.0, 0.0, peptide_weight=5.0,
                          lambda_sc_fape=0.0, lambda_chi=0.0,
                          plddt_peptide_ratio=(a.plddt_peptide_ratio
                                               if a.plddt_peptide_ratio > 0 else None)).to(dev)

    idx = pd.read_csv(Path(a.h5_dir) / "index.csv", dtype=str).head(a.n * 3)
    ds = m1.H5DistillDataset(idx["id"].tolist(), dict(zip(idx["id"], idx["shard"])), a.h5_dir)
    loader = m1.make_dataloader(ds, 1, shuffle=False, num_workers=2)

    # Which PATH does a loss actually train? The SM builds out["single"] from s
    # conditioned on z (the pair rep enters as the IPA's attention bias), so a loss on
    # the SM's single output does NOT obviously train the single path — it could flow
    # predominantly into pair. That matters: the point of the pLDDT term is to improve
    # the SINGLE representation the frozen head reads.
    SINGLE = ("s_proj", "sm_s", "ipas", "self_attn", "s_norm", "attn_norm")
    PAIR = ("z_proj", "sm_z", "opm", "tri_out", "tri_in", "pair_trans")

    def split(name):
        if any(t in name for t in PAIR):   return "pair"
        if any(t in name for t in SINGLE): return "single"
        return "other"

    def gnorm(loss):
        net.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        tot = {"single": 0.0, "pair": 0.0, "other": 0.0}
        for n_, prm in net.named_parameters():
            if prm.grad is not None and prm.requires_grad:
                tot[split(n_)] += float(prm.grad.norm()) ** 2
        allsq = sum(tot.values())
        return allsq ** 0.5, {k: v ** 0.5 for k, v in tot.items()}

    gs, gp, split_s, split_p = [], [], [], []
    for i, batch in enumerate(loader):
        if i >= a.n:
            break
        batch = m1.move_batch(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = net(batch, return_frames=True)
            ca, pl, pae, fr = o[:4]; aux = o[4]
            ls, _ = struct(ca, pl, pae, fr, batch, aux=aux)
            lc, _ = conf(ca, pl, pae, fr, batch, aux=aux)
        a_, sa = gnorm(ls); b_, sb = gnorm(lc)
        gs.append(a_); gp.append(b_); split_s.append(sa); split_p.append(sb)

    gs, gp = np.array(gs), np.array(gp)
    print(f"\nckpt {a.ckpt}   n={len(gs)} real batches\n")
    print(f"  ||g_struct|| : median {np.median(gs):8.3f}   p90 {np.percentile(gs,90):8.3f}")
    print(f"  ||g_plddt||  : median {np.median(gp):8.3f}   p90 {np.percentile(gp,90):8.3f}")
    ratio = np.median(gp) / max(np.median(gs), 1e-9)
    print(f"  ratio (plddt/struct) at lambda=1 : {ratio:8.2f}x")
    import numpy as _np
    print("\n  WHERE each loss puts its gradient (median over batches):")
    print(f"  {'loss':<10} {'single path':>13} {'pair path':>12} {'other':>9}   single share")
    for nm, sp in (("structure", split_s), ("pLDDT CE", split_p)):
        med = {k: _np.median([d[k] for d in sp]) for k in ("single", "pair", "other")}
        tot = sum(med.values()) or 1e-9
        print(f"  {nm:<10} {med['single']:>13.3f} {med['pair']:>12.3f} {med['other']:>9.3f}"
              f"   {med['single']/tot*100:>8.1f}%")
    print(f"\n  lambda for confidence grad = {a.target_frac:.0%} of structural:"
          f"  {a.target_frac/max(ratio,1e-9):.4g}")
    print(f"  at the inherited lambda=0.01, confidence is "
          f"{0.01*ratio*100:.1f}% of the structural gradient")
    if 0.01 * ratio > 0.5:
        print("  => 0.01 is TOO STRONG on this path; it would dominate the structure loss.")
    elif 0.01 * ratio < 0.02:
        print("  => 0.01 is very weak here; confidence would barely learn.")
    else:
        print("  => 0.01 is in a reasonable range.")


if __name__ == "__main__":
    main()
