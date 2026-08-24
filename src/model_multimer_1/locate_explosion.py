"""WHERE in the network does the gradient explode?

The per-module table says "trunk", but that is 92% of the trainable parameters and it
cannot separate two very different stories:
  (a) the trunk GENERATES the large gradient itself, or
  (b) a modest gradient at the output is AMPLIFIED backwards through the 8 frozen IPA
      layers of the StructureModule before it ever reaches the trunk.
The SM is frozen, so it has no parameter gradients — but activations still carry the
backward signal through it, and that is measurable with tensor hooks.

Measures, on ONE batch, for a given checkpoint:
  FORWARD  : activation norm at each stage (embedder -> head2 -> each trunk block ->
             sm_s/sm_z -> SM single/frames -> angles -> atom14)
  BACKWARD : grad-wrt-ACTIVATION norm at the same points, so you can read the
             amplification factor from one stage to the next.

    $PY src/model_multimer_1/locate_explosion.py --ckpt <ckpt> --angle-input raw|layernorm
"""
from __future__ import annotations
import argparse, sys, glob
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
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm")
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--n", type=int, default=40, help="batches to scan; the WORST is reported")
    p.add_argument("--n-trunk", type=int, default=3)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = MM.MultimerModel(n_trunk=a.n_trunk, device=dev, pep_frames="identity",
                           angle_input=a.angle_input)
    net.set_stage(1)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)["trainable"]
    net.load_state_dict(sd, strict=False)
    net.train()
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                              lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)

    idx = pd.read_csv(Path(a.h5_dir) / "index.csv", dtype=str).head(a.n * 4)
    ds = m1.H5DistillDataset(idx["id"].tolist(), dict(zip(idx["id"], idx["shard"])), a.h5_dir)
    loader = m1.make_dataloader(ds, 1, shuffle=False, num_workers=2)

    best = None
    for i, batch in enumerate(loader):
        if i >= a.n:
            break
        batch = m1.move_batch(batch, dev)
        taps = {}

        def tap(name, t):
            """record forward norm now, and grad-wrt-this-activation at backward"""
            if not torch.is_tensor(t) or not t.requires_grad:
                return t
            taps[name] = {"fwd": float(t.detach().float().norm())}
            t.register_hook(lambda g, n=name: taps[n].__setitem__(
                "bwd", float(g.detach().float().norm())))
            return t

        net.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            f = MM.build_multimer_feats(batch["aatype"], batch["segment_id"], batch["seq_mask"])
            msa_emb, z = net.embedder(f)
            s = tap("1_embedder_single", msa_emb[..., 0, :, :])
            z = tap("2_embedder_pair", z)
            idm = m1.peptide_mask_from_batch(batch["seq_mask"], batch["segment_id"])
            frames = MM._frames_from_bb(batch["teacher_bb"], batch["seq_mask"], 0.0, idm)
            h2s, h2p = net.head2(batch["teacher_bb"], f["mhc"], frames)
            h2pool = (h2s * f["mhc"][..., None]).sum(1, keepdim=True) / \
                f["mhc"].sum(1, keepdim=True).clamp_min(1.0)[..., None]
            h2pool = h2pool.expand(-1, s.shape[1], -1)
            anc = torch.nn.functional.one_hot(batch["anchor"].clamp(0, 1).long(), 2).float()
            ancp = torch.maximum(anc[:, :, None], anc[:, None, :])
            s = tap("3_s_proj", net.s_proj(torch.cat([s, h2pool, anc], -1)))
            z = tap("4_z_proj", net.z_proj(torch.cat([z, h2p, ancp], -1)))
            mask = batch["seq_mask"]
            for bi, blk in enumerate(net.trunk):
                s, z = blk(s, z, frames, mask)
                s = tap(f"5_trunk{bi}_single", s); z = tap(f"5_trunk{bi}_pair", z)
            sm_in = tap("6_sm_s_INPUT_TO_FROZEN_SM", net.sm_s(s))
            sm_z = tap("6_sm_z", net.sm_z(z))
            out = net.sm({"single": sm_in, "pair": sm_z}, batch["aatype"], mask=mask)
            sm_single = tap("7_SM_OUTPUT_single", out["single"])
            s_init = (net.sm.layer_norm_s(sm_in) if net.angle_input == "layernorm" else sm_in)
            un, ang = net.angle_head(sm_single, s_init)
            ang = tap("8_angles", ang)
            bb = MM.Rigid3Array.from_array4x4(out["frames"][-1])
            allf = net.sm.torsion_angles_to_frames(bb, ang, batch["aatype"])
            atom14 = tap("9_atom14", net.sm.frames_and_literature_positions_to_atom14_pos(
                allf, batch["aatype"]))
            aux = {"angles": ang, "unnormalized_angles": un,
                   "sidechain_frames": allf.to_tensor_4x4(), "atom14": atom14}
            total, _ = loss_mod(atom14[..., 1, :], net.plddt(net.plddt_proj(sm_single)),
                                None, out["frames"], batch, aux=aux)
        total.backward()
        g = sum(float(p.grad.norm()) ** 2 for p in net.trainable_parameters()
                if p.grad is not None) ** 0.5
        if best is None or g > best[0]:
            best = (g, batch.get("id", [f"batch{i}"])[0], dict(taps))

    g, sid, taps = best
    print(f"\ncheckpoint: {a.ckpt}  (angle_input={a.angle_input})")
    print(f"WORST of {a.n} batches: |g|_params = {g:.1f}   structure = {sid}\n")
    print(f"{'stage':<34} {'||activation||':>15} {'||grad wrt act||':>17} {'amplification':>14}")
    prev = None
    for k in sorted(taps):
        d = taps[k]
        b = d.get("bwd", float("nan"))
        amp = (b / prev) if (prev and prev > 0 and b == b) else float("nan")
        print(f"{k:<34} {d['fwd']:>15.2f} {b:>17.4g} "
              f"{'' if amp != amp else f'{amp:>13.2f}x'}")
        if b == b:
            prev = b
    print("\nRead BACKWARD (bottom -> top): the row where 'amplification' jumps is where")
    print("the gradient grows. Rows 7->6 spanning the FROZEN StructureModule show whether")
    print("its 8 IPA layers amplify on the way back to the trunk.")


if __name__ == "__main__":
    main()
