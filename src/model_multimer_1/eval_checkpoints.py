"""Evaluate one or more checkpoints on the SAME held-out sets, so they are comparable.

Training only ever reports validation at an EPOCH boundary. Every full-dataset run so
far died mid-epoch-1, so their checkpoints have never been scored on anything — the only
held-out numbers in the project come from a 6,000-structure short run. This scores any
checkpoint on the real two_axis val fold (and val-matched, the confidence-filtered
subset that is the apples-to-apples comparison against a filtered-train stage).

    $PY src/model_multimer_1/eval_checkpoints.py --ckpts a.pt b.pt --max-val 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import train as T                                                     # noqa: E402
import model as MM                                                    # noqa: E402
sys.path.insert(0, str(_HERE.parent / "model"))
import utils as m1                                                    # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--max-val", type=int, default=3000)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--hasmig-dir", default="data/processed/h5_store_hasmig")
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm",
                   help="must match how the checkpoint was TRAINED. Checkpoints from "
                        "before commit 7fa2fa4 were trained with 'raw'; scoring them "
                        "under 'layernorm' penalises them for a convention they never "
                        "saw, so compare like with like.")
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = T.parse_args(["--stage", "1"])        # --stage is required; then override
    args.sidechains = True
    # parse_args defaults --h5-dir to data/processed/h5_store, which does NOT exist —
    # build_datasets then silently yields an EMPTY val set and every metric prints "-".
    # Pin the stores the runs actually used, and refuse to score nothing.
    args.h5_dir = a.h5_dir
    args.hasmig_dir = a.hasmig_dir
    args.max_val, args.bs, args.num_workers = a.max_val, a.bs, a.num_workers
    args.n_trunk, args.pep_frames = a.n_trunk, "identity"

    _, val_ds, val_hi = T.build_datasets(args, filt=True, hasmig_w=1.0)
    import random
    from torch.utils.data import Subset

    def cap(ds, seed):
        if ds is None or len(ds) <= a.max_val:
            return ds
        j = list(range(len(ds)))
        random.Random(seed).shuffle(j)
        return Subset(ds, sorted(j[:a.max_val]))
    val_ds, val_hi = cap(val_ds, args.seed + 1), cap(val_hi, args.seed + 2)
    n_v = len(val_ds) if val_ds else 0
    n_vm = len(val_hi) if val_hi else 0
    print(f"val={n_v}  val-matched={n_vm}  (identical sets for every checkpoint)\n")
    if n_v == 0:
        raise SystemExit(
            f"FATAL: empty validation set from --h5-dir '{args.h5_dir}'. Scoring would "
            f"print '-' for every metric and look like a model failure rather than a "
            f"path error. Point --h5-dir at the store the runs used.")

    # SAME loss config the stage-1 runs used, so `total` is comparable to their logs
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=args.peptide_weight,
                              lambda_sc_fape=args.sc_fape_w,
                              lambda_chi=args.chi_w).to(dev)
    def build_for(sd):
        """Build the model to MATCH the checkpoint, detected from its own keys.

        Scoring a checkpoint through an architecture it was never trained with is
        silently wrong, not an error — it just reports a worse model. That already
        happened once here: checkpoints predating the pre-attention LayerNorm were
        scored WITH it and came out understated. Flags are easy to forget; the keys
        are ground truth.
          attn_norm.*   present  <=> the trunk had the pre-attention LayerNorm
          plddt_proj.*  present  <=> the trainable pLDDT adapter was used
        (plddt_proj is frozen in stage 1 so it is absent from stage-1 checkpoints —
        which is correct, since frozen identity-init == nn.Identity.)
        """
        attn = any("attn_norm" in k for k in sd)
        adapter = any("plddt_proj" in k for k in sd)
        m = MM.MultimerModel(n_trunk=a.n_trunk, device=dev, pep_frames="identity",
                             angle_input=a.angle_input, attn_norm=attn,
                             plddt_adapter=adapter)
        m.set_stage(1)
        return m, attn, adapter

    hdr = f"{'checkpoint':38} {'gstep':>7} | {'val tot':>8} {'pepRMSD':>8} {'pLDDT-MAE':>9} " \
          f"| {'vm tot':>7} {'pepRMSD':>8} {'pLDDT-MAE':>9}"
    print(hdr); print("-" * len(hdr))
    for c in a.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        sd = ck.get("trainable", ck.get("model", ck))
        bad = [k for k, v in sd.items()
               if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            print(f"{Path(c).parent.name+'/'+Path(c).name:38} {'--':>7} | "
                  f"NON-FINITE ({len(bad)}/{len(sd)} tensors) — dead checkpoint")
            continue
        net, _attn, _ad = build_for(sd)
        missing = net.load_state_dict(sd, strict=False).missing_keys
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        lost = [k for k in missing if k in trainable]
        if lost:
            print(f"  WARNING {Path(c).name}: {len(lost)} TRAINABLE tensors missing from "
                  f"the checkpoint (left at random init): {lost[:4]}")
        net.eval()
        out = []
        for ds in (val_ds, val_hi):
            if ds is None or not len(ds):
                out.append(None); continue
            ld = m1.make_dataloader(ds, a.bs, shuffle=False, num_workers=a.num_workers)
            out.append(m1.evaluate(net, ld, loss_mod, dev))
        v, vm = out
        name = Path(c).parent.name + "/" + Path(c).name
        name += f" [{'LN' if _attn else '--'}{'/ad' if _ad else ''}]"
        row = f"{name:38} {ck.get('global_step','?'):>7} | "
        row += (f"{v['total']:8.3f} {v['pep_ca_rmsd']:8.2f} {v['pep_plddt_mae']:9.2f} | "
                if v else f"{'-':>8} {'-':>8} {'-':>9} | ")
        row += (f"{vm['total']:7.3f} {vm['pep_ca_rmsd']:8.2f} {vm['pep_plddt_mae']:9.2f}"
                if vm else f"{'-':>7} {'-':>8} {'-':>9}")
        print(row)


if __name__ == "__main__":
    main()
