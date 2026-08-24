"""WHY does the model collapse under ordinary gradients?

Observed: in run 29383715 (spike rejection ON) ca_rmsd climbed 0.59 -> 1.39 A while
|g|max stayed at 5-6. The behavioural collapse PRECEDES any gradient explosion, and
what breaks includes the MHC — which is handed to the model as an INPUT. For the
easiest part of the task to fail, the frozen StructureModule must be receiving a
representation it can no longer decode.

Structural reason this is possible at all: the SM is FROZEN and starts from
Rigid3Array.identity (black-hole init, structure_module.py:1100). It must regenerate
all ~190 residues from `sm_s(trunk_single)` alone. The encoder's only lever on an
8-layer frozen decoder is that input vector, and NOTHING constrains it to stay in the
distribution the SM was pretrained on. Push it outside, and the decoder falls off a
cliff that the loss gradient gives no warning about.

Two measurements per checkpoint, on ONE fixed batch of real structures:

  A. SM-INPUT DRIFT — statistics of sm_s(s) (pre- and post-LayerNorm) and sm_z(z).
     LayerNorm absorbs global scale, so the informative quantity is the cosine
     similarity of the NORMALISED input against the last known-stable checkpoint:
     if that falls, the frozen decoder is being fed a different kind of vector.

  B. SHARPNESS — perturb every trainable weight by a relative sigma and re-measure the
     loss. A sharp region means an ORDINARY-sized optimiser step produces a large loss
     increase, which is exactly "normal gradients, sudden collapse". If the
     pre-collapse checkpoint is much sharper than the stable one, the LR is too high
     FOR THAT REGION and no amount of gradient-spike rejection can help.

    $PY src/model_multimer_1/diagnose_collapse.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "openfold"))
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_HERE))
import utils as m1                                                    # noqa: E402
import model as MM                                                    # noqa: E402


def load_batch(dev, n=6, h5_dir="data/processed/h5_store_sc"):
    """One fixed batch of real structures (identical for every checkpoint).

    Uses H5DistillDataset + collate_with_teacher — the SAME path training uses — rather
    than hand-assembling tensors, which silently omits fields collate_fn requires.
    """
    idx = pd.read_csv(Path(h5_dir) / "index.csv", dtype={"id": str, "shard": str})
    id_to_shard = dict(zip(idx["id"], idx["shard"]))
    ids = idx["id"].tolist()[:n]
    ds = m1.H5DistillDataset(ids, id_to_shard, Path(h5_dir))
    exs = [ds[i] for i in range(len(ds))]
    return m1.move_batch(m1.collate_with_teacher(exs), dev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-struct", type=int, default=6)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="label=path pairs to override the built-in list. The FIRST is "
                        "the reference the cosine similarity is measured against, so put "
                        "the known-stable checkpoint first.")
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batch = load_batch(dev, args.n_struct)
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                              lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)

    cks = [("short g9000  STABLE", "checkpoints_mm1/mm1_s1_short/last.pt"),
           ("full  g29600 pre-collapse", "checkpoints_mm1/snapshots/s1_full_mid.pt"),
           ("full  g52818 degrading", "checkpoints_mm1/snapshots/s1_full_ep2_g52818.pt"),
           ("full  g54000 dying", "checkpoints_mm1/mm1_s1_full/last.pt")]
    if args.ckpts:
        cks = [tuple(x.split("=", 1)) for x in args.ckpts]

    net = MM.MultimerModel(n_trunk=3, device=dev, pep_frames="identity",
                           angle_input=args.angle_input)
    net.set_stage(1)
    ref_dir = None
    print(f"batch: {args.n_struct} real structures, N={batch['aatype'].shape[1]}\n")
    print("=== A. WHAT THE FROZEN STRUCTURE MODULE IS BEING FED ===")
    print("  ||raw|| is what the angle head saw BEFORE the fix (--angle-input raw);")
    print("  ||LN||  is what AlphaFold feeds it, and what it gets now (default).")
    print(f"{'checkpoint':<27} {'||raw||':>10} {'||LN||':>9} {'std':>8} "
          f"{'cos vs stable':>14} {'loss':>8}")
    rows = []
    for name, f in cks:
        if not Path(f).exists():
            continue
        sd = torch.load(f, map_location="cpu", weights_only=False)["trainable"]
        net.load_state_dict(sd, strict=False)
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=dev == "cuda"):
            s, z, mask = net._encode(batch)
            sm_in = net.sm_s(s).float()
            ln = net.sm.layer_norm_s(sm_in)            # what the SM actually consumes
            o = net(batch, return_frames=True)         # o[0] = predicted CA (atom14[:,1])
            ca = o[0].float()
            total, terms = loss_mod(o[0], o[1], o[2], o[3], batch, aux=o[4])
        m = batch["seq_mask"].bool()
        d = ln[m].flatten().float()
        if ref_dir is None:
            ref_dir = d.clone()
        cos = float(torch.nn.functional.cosine_similarity(d[None], ref_dir[None])) \
            if d.numel() == ref_dir.numel() else float("nan")
        print(f"{name:<27} {float(sm_in.norm()):>10.1f} {float(ln.float().norm()):>9.1f} "
              f"{float(sm_in.std()):>8.3f} {cos:>14.4f} {float(total):>8.3f}")
        rows.append((name, f))

    print("\n=== B. SHARPNESS: loss after perturbing EVERY trainable weight by sigma*||W|| ===")
    print("    (a sharp region = an ORDINARY step causes a large loss jump)")
    sigmas = [0.001, 0.003, 0.01, 0.03]
    print(f"{'checkpoint':<27} {'base':>8} " + " ".join(f"{'s=' + str(s):>9}" for s in sigmas))
    for name, f in rows:
        sd = torch.load(f, map_location="cpu", weights_only=False)["trainable"]
        net.load_state_dict(sd, strict=False)
        net.eval()
        clean = {k: v.detach().clone() for k, v in net.named_parameters() if v.requires_grad}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=dev == "cuda"):
            o = net(batch, return_frames=True)
            base = float(loss_mod(o[0], o[1], o[2], o[3], batch, aux=o[4])[0])
        line = f"{name:<27} {base:>8.3f} "
        for sg in sigmas:
            vals = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                with torch.no_grad():
                    for k, v in net.named_parameters():
                        if k in clean:
                            v.copy_(clean[k] + sg * clean[k].norm() /
                                    max(clean[k].numel() ** 0.5, 1) * torch.randn_like(clean[k]))
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        o = net(batch, return_frames=True)
                        vals.append(float(loss_mod(o[0], o[1], o[2], o[3], batch, aux=o[4])[0]))
            with torch.no_grad():
                for k, v in net.named_parameters():
                    if k in clean:
                        v.copy_(clean[k])
            line += f"{np.mean(vals):>9.3f} "
        print(line)
    print("\ninterpretation: if the pre-collapse checkpoint's loss rises far more steeply")
    print("with sigma than the stable one, the model has moved into a SHARP region and a")
    print("normal-sized step at lr 5e-4 is enough to throw it out — no gradient spike needed.")


if __name__ == "__main__":
    main()
