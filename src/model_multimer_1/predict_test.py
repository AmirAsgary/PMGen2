"""
Run a model_multimer_1 checkpoint on the 15 class-I examples in data/test/, write
FULL-ATOM PDBs, superpose each prediction onto its reference PDB, and report RMSDs.

Superposition convention (the pMHC binding-pose convention): the prediction is
Kabsch-superposed onto the reference using the **MHC Cα** atoms, then RMSDs are
measured. The written PDB is in the reference frame, so it overlays directly.

  --pep-frames teacher   the peptide's TRUE backbone frames are fed to the trunk
                         (what every checkpoint so far was trained with — this LEAKS
                         the ground-truth peptide pose)
  --pep-frames identity  peptide frames withheld (the documented design): the peptide
                         pose must actually be predicted

  $PY src/model_multimer_1/predict_test.py --ckpt <last.pt> --tag A \
      --pep-frames teacher --out-dir outputs/mm1_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "openfold"))
sys.path.insert(0, str(_ROOT / "src" / "model"))
sys.path.insert(0, str(_HERE))

import utils as m1                                                   # noqa: E402
import model as MM                                                   # noqa: E402
from openfold.np import protein as ofp, residue_constants as rc      # noqa: E402
from openfold.utils.feats import atom14_to_atom37                    # noqa: E402
from openfold.data.data_transforms import make_atom14_masks          # noqa: E402


def kabsch(mob: np.ndarray, ref: np.ndarray):
    """Rotation+translation mapping `mob` onto `ref` (both [K,3], correspondence)."""
    mu_m, mu_r = mob.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((mob - mu_m).T @ (ref - mu_r))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, mu_m, mu_r


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean())) if len(a) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True, help="label for outputs, e.g. A / B")
    p.add_argument("--pep-frames", choices=["teacher", "identity"], default="identity",
                   help="'identity' (DEFAULT, leak-free) withholds the peptide's true "
                        "backbone frames. 'teacher' LEAKS the ground-truth pose and only "
                        "exists to reproduce the pre-fix checkpoints.")
    p.add_argument("--out-dir", default="outputs/mm1_test")
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--leak-test", action="store_true",
                   help="per example, ALSO predict with the peptide's teacher_bb shifted "
                        "+10 A and the side-chain/confidence targets randomised, and report "
                        "the max change in the predicted peptide atoms. 0.0 == no leakage "
                        "on these actual test structures (not just synthetic ones).")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) / f"{args.tag}_{args.pep_frames}"
    out_dir.mkdir(parents=True, exist_ok=True)

    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames=args.pep_frames)
    net.set_stage(args.stage)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["trainable"], strict=False)
    net.eval()

    rows = m1.dummy_rows()
    ds = m1.DistillDataset(rows)
    recs = []
    for i, r in enumerate(rows):
        ex = ds[i]
        b = m1.move_batch(m1.collate_with_teacher([ex]), dev)
        pep_t = m1.peptide_mask_from_batch(b["seq_mask"], b["segment_id"]).bool()[0]
        mhc_t = b["seq_mask"].bool()[0] & ~pep_t
        pep, mhc = pep_t.cpu().numpy(), mhc_t.cpu().numpy()

        out = net.predict(b)
        aatype = b["aatype"]
        if args.leak_test:
            # `frames` is the only path from teacher_bb to the peptide; the targets must
            # never reach the model at all. Perturb both and require a bit-identical
            # peptide prediction.
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b2["teacher_bb"][0][pep_t] += torch.tensor([10.0, 0.0, 0.0], device=dev)
            for key in ("teacher_atom14", "teacher_chi", "teacher_ca", "teacher_plddt",
                        "teacher_pae"):
                if key in b2 and torch.is_tensor(b2[key]):
                    b2[key] = torch.randn_like(b2[key]) * 10.0
            d = (net.predict(b2)["atom14"] - out["atom14"])[0][pep_t]
            leak_delta = float(d.abs().max()) if d.numel() else 0.0
        else:
            leak_delta = float("nan")
        masks = make_atom14_masks({"aatype": aatype})
        atom37 = atom14_to_atom37(out["atom14"], masks)[0].float().cpu().numpy()  # [N,37,3]
        pred_mask = masks["atom37_atom_exists"][0].cpu().numpy()                  # [N,37]
        plddt = m1.plddt_from_logits(out["plddt_logits"],
                                     net.plddt_no_bins if hasattr(net, "plddt_no_bins")
                                     else 50)[0].float().cpu().numpy()            # [N] 0-100

        # ---- reference (the AlphaFold/PMGen structure this example was parsed from)
        pdb_path, _, _ = m1.find_teacher_files(Path(r["alphafold_output_path"]))
        ref = ofp.from_pdb_string(pdb_path.read_text())
        ref_xyz, ref_mask = ref.atom_positions, ref.atom_mask                     # [N,37,*]
        assert ref_xyz.shape[0] == atom37.shape[0], "residue count mismatch"

        # ---- superpose prediction onto reference using MHC Cα, apply to ALL atoms
        ca = rc.atom_order["CA"]
        rot, mu_m, mu_r = kabsch(atom37[mhc, ca, :], ref_xyz[mhc, ca, :])
        pred_sup = (atom37 - mu_m) @ rot.T + mu_r

        both = (pred_mask * ref_mask).astype(bool)                                # [N,37]
        bb_idx = [rc.atom_order[a] for a in ("N", "CA", "C")]

        def sel(m_res, atoms=None):
            m = np.zeros_like(both)
            m[m_res] = both[m_res]
            if atoms is not None:
                keep = np.zeros(37, bool); keep[atoms] = True
                m &= keep[None, :]
            return pred_sup[m], ref_xyz[m]

        rec = {"id": r["id"], "n_pep": int(pep.sum()), "n_mhc": int(mhc.sum())}
        rec["pep_ca_rmsd"] = rmsd(*sel(pep, [ca]))
        rec["pep_bb_rmsd"] = rmsd(*sel(pep, bb_idx))
        rec["pep_allatom_rmsd"] = rmsd(*sel(pep))
        rec["mhc_ca_rmsd"] = rmsd(*sel(mhc, [ca]))
        rec["all_ca_rmsd"] = rmsd(*sel(np.ones_like(pep, bool), [ca]))
        rec["all_atom_rmsd"] = rmsd(*sel(np.ones_like(pep, bool)))
        rec["pep_plddt_pred"] = float(plddt[pep].mean())
        rec["pep_plddt_teacher"] = float(b["teacher_plddt"][0][pep_t].mean())
        rec["anchors"] = r.get("anchors", "")
        rec["leak_delta_A"] = leak_delta
        recs.append(rec)

        # ---- write the superposed full-atom prediction
        bf = np.repeat(plddt[:, None], 37, axis=1) * pred_mask
        chain_index = np.where(pep, 1, 0)          # 0 = MHC (A), 1 = peptide (B)
        prot = ofp.Protein(atom_positions=pred_sup, aatype=aatype[0].cpu().numpy(),
                           atom_mask=pred_mask, residue_index=np.arange(1, len(pep) + 1),
                           b_factors=bf, chain_index=chain_index)
        (out_dir / f"{r['id']}_pred.pdb").write_text(ofp.to_pdb(prot))

    import pandas as pd
    df = pd.DataFrame(recs)
    csv = out_dir / "rmsd.csv"
    df.to_csv(csv, index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== {args.tag}  pep_frames={args.pep_frames}  ({len(df)} structures) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nMEAN " + "  ".join(
        f"{c}={df[c].mean():.3f}" for c in
        ["pep_ca_rmsd", "pep_bb_rmsd", "pep_allatom_rmsd", "mhc_ca_rmsd",
         "all_ca_rmsd", "all_atom_rmsd", "pep_plddt_pred", "pep_plddt_teacher"]))
    if args.leak_test:
        mx = float(df["leak_delta_A"].max())
        print(f"\nLEAK TEST on these {len(df)} structures: max |Δ predicted peptide atom| "
              f"when the peptide's teacher_bb is shifted +10 A and every target is "
              f"randomised = {mx:.3e} A")
        print("  PASS: no leakage — the peptide pose is predicted, not read."
              if mx == 0.0 else
              f"  *** FAIL: the prediction moved {mx:.4f} A — ground-truth peptide "
              f"information is reaching the model. ***")
    print(f"\nPDBs + {csv.name} -> {out_dir}")


if __name__ == "__main__":
    main()
