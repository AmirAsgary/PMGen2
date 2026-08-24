"""Does predicted pLDDT track the truth ON THE PEPTIDE — and does it predict ERROR?

The training metric `plddt_spearman` pools ALL residues, and ~95% of a pMHC is MHC whose
confidence is easy and nearly constant. That swamps the peptide, which is the part we
actually predict and the part a user needs a trust signal for.

Three separate questions, reported separately because they are not the same thing:

  A. per-RESIDUE, peptide only : does it rank residues within/across peptides correctly?
  B. per-STRUCTURE mean        : does it rank whole predictions by teacher confidence?
  C. pred pLDDT vs ACTUAL Ca-RMSD (per structure) — THE PRACTICAL ONE. A screening tool
     needs "trust this prediction" and that means anti-correlating with real error.
     B can look fine while C is useless, since matching the teacher's confidence is not
     the same as knowing when you are wrong.

    $PY src/model_multimer_1/plddt_corr.py --ckpts a.pt b.pt --n 300
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "openfold"))
sys.path.insert(0, str(_HERE.parents[1] / "src" / "model"))
sys.path.insert(0, str(_HERE))
# ORDER MATTERS. model_multimer_1/model.py does its own
# `sys.path.insert(0, src/model)` at import time, which puts src/model AHEAD of this
# directory — and src/model has its OWN train.py. So `train` must be imported BEFORE
# `model`, or `import train` silently resolves to the wrong module (it did: the job
# failed with "unrecognized arguments: --stage 1", model_1's parser).
import train as T                                                     # noqa: E402
assert "model_multimer_1" in T.__file__, f"wrong train module: {T.__file__}"
import model as MM                                                    # noqa: E402
import utils as m1                                                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300, help="val structures")
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm")
    p.add_argument("--filtered", action="store_true",
                   help="use the confidence-FILTERED val subset instead of the full fold")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    args = T.parse_args(["--stage", "1"])
    args.sidechains = True
    args.h5_dir = "data/processed/h5_store_sc"
    args.hasmig_dir = "data/processed/h5_store_hasmig"
    args.max_val = a.n
    _, val_full, val_hi = T.build_datasets(args, filt=True, hasmig_w=1.0)
    ds = val_hi if a.filtered else val_full
    import random
    from torch.utils.data import Subset
    if ds is not None and len(ds) > a.n:
        j = list(range(len(ds))); random.Random(1).shuffle(j)
        ds = Subset(ds, sorted(j[:a.n]))
    print(f"val subset: {len(ds)} structures ({'FILTERED' if a.filtered else 'full fold'})\n")

    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0).to(dev)
    for c in a.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        sd = ck.get("trainable", ck)
        net = MM.MultimerModel(n_trunk=a.n_trunk, device=dev, pep_frames="identity",
                               angle_input=a.angle_input,
                               attn_norm=any("attn_norm" in k for k in sd),
                               plddt_adapter=any("plddt_proj" in k for k in sd))
        net.set_stage(1); net.load_state_dict(sd, strict=False); net.eval()

        res_p, res_t, st_p, st_t, st_err = [], [], [], [], []
        ld = m1.make_dataloader(ds, 1, shuffle=False, num_workers=4)
        with torch.no_grad():
            for b in ld:
                b = m1.move_batch(b, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = net.predict(b)
                pl = m1.plddt_from_logits(o["plddt_logits"].float(), 50)[0].cpu().numpy()
                tp = b["teacher_plddt"][0].float().cpu().numpy()
                pep = m1.peptide_mask_from_batch(b["seq_mask"], b["segment_id"])[0].bool().cpu().numpy()
                mhc = b["seq_mask"][0].bool().cpu().numpy() & ~pep
                res_p.append(pl[pep]); res_t.append(tp[pep])
                st_p.append(pl[pep].mean()); st_t.append(tp[pep].mean())
                # actual peptide error, MHC-superposed
                pc = o["ca"][0].float().cpu().numpy().astype(np.float64)
                tc = b["teacher_ca"][0].float().cpu().numpy().astype(np.float64)
                mc, rc = pc[mhc].mean(0), tc[mhc].mean(0)
                P, Q = pc[mhc]-mc, tc[mhc]-rc
                U, S, Vt = np.linalg.svd(P.T@Q); d = np.sign(np.linalg.det(Vt.T@U.T))
                R = Vt.T@np.diag([1,1,d])@U.T
                st_err.append(float(np.sqrt(((((pc-mc)@R.T)[pep]-(tc-rc)[pep])**2).sum(1).mean())))
        rp, rt = np.concatenate(res_p), np.concatenate(res_t)
        sp, st_, se = np.array(st_p), np.array(st_t), np.array(st_err)
        name = Path(c).parent.name + "/" + Path(c).name
        print(f"### {name}  (gstep {ck.get('global_step','?')}, stage {ck.get('stage','?')})")
        print(f"  A. per-RESIDUE (peptide only, n={len(rp)})")
        print(f"       pred vs teacher pLDDT : Pearson {pearsonr(rp,rt)[0]:+.3f}   Spearman {spearmanr(rp,rt).statistic:+.3f}")
        print(f"  B. per-STRUCTURE mean peptide pLDDT (n={len(sp)})")
        print(f"       pred vs teacher pLDDT : Pearson {pearsonr(sp,st_)[0]:+.3f}   Spearman {spearmanr(sp,st_).statistic:+.3f}")
        print(f"  C. per-STRUCTURE pred pLDDT vs ACTUAL peptide Ca-RMSD  <-- the practical one")
        print(f"       Pearson {pearsonr(sp,se)[0]:+.3f}   Spearman {spearmanr(sp,se).statistic:+.3f}"
              f"   (want NEGATIVE: high confidence -> low error)")
        print(f"       pred pLDDT range {sp.min():.1f}-{sp.max():.1f} (spread {sp.max()-sp.min():.1f})"
              f" | true RMSD range {se.min():.2f}-{se.max():.2f} A")
        print(f"       MAE(pred,teacher) = {np.abs(sp-st_).mean():.2f}\n")


if __name__ == "__main__":
    main()
