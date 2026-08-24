"""Benchmark model_multimer_1 against the published pRMSD benchmark.

For every complex in ``pRMSD_benchmark.tsv`` the model is run once per anchor
hypothesis, the variant with the highest mean PEPTIDE pLDDT is kept, and the
peptide is scored against the crystal structure after superposing on the MHC
only -- the pMHC binding-pose convention the benchmark's other methods use.

Anchor hypotheses (1-indexed, L = peptide length):
    L >= 9 : (1,L) (1,L-1) (1,L-2) (2,L) (2,L-1) (3,L)
    L == 8 : (1,L) (1,L-1) (2,L)

The MHC backbone is an INPUT to this model (that is the PMGen design: the MHC
structure is given and refined on peptide binding), so --mhc-source selects
where it comes from:

    crystal   the target's own crystal MHC.  Upper bound: the groove has already
              relaxed around the true peptide, which is information the
              sequence-only baselines do not get.
    template  the MHC of a DIFFERENT benchmark complex -- the nearest neighbour
              by MHC sequence identity that carries a different peptide, spliced
              onto this target after MHC-Calpha superposition.  This is the
              deployment case and the honest comparison.

The peptide's own backbone is withheld in both cases (--pep-frames identity);
--leak-test re-runs each kept prediction with the peptide's teacher frames
displaced +10 A and every target randomised, and requires a bit-identical answer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "openfold"))
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "model_multimer_1"))

import utils as m1                                                    # noqa: E402
import model as MM                                                    # noqa: E402
from openfold.np import protein as ofp, residue_constants as rc       # noqa: E402
from openfold.utils.feats import atom14_to_atom37                     # noqa: E402
from openfold.data.data_transforms import make_atom14_masks           # noqa: E402


def anchor_sets(L: int) -> list[str]:
    """The anchor hypotheses to enumerate for a peptide of length L."""
    if L == 8:
        pairs = [(1, L), (1, L - 1), (2, L)]
    else:
        pairs = [(1, L), (1, L - 1), (1, L - 2), (2, L), (2, L - 1), (3, L)]
    return [f"{a};{b}" for a, b in pairs]


def kabsch(mob: np.ndarray, ref: np.ndarray):
    mu_m, mu_r = mob.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((mob - mu_m).T @ (ref - mu_r))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, mu_m, mu_r


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean())) if len(a) else float("nan")


def seq_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(x == y for x, y in zip(a[:n], b[:n]))
    return same / max(len(a), len(b))


def pick_templates(bm: pd.DataFrame) -> dict:
    """For each target, the most MHC-similar OTHER complex with a different peptide."""
    tpl = {}
    seqs = bm.mhc_seq.tolist(); peps = bm.peptide.tolist(); pdbs = bm.PDB.tolist()
    for i in range(len(bm)):
        best, best_id = -1.0, None
        for j in range(len(bm)):
            if i == j or peps[j] == peps[i]:
                continue
            s = seq_identity(seqs[i], seqs[j])
            if s > best:
                best, best_id = s, pdbs[j]
        tpl[pdbs[i]] = (best_id, best)
    return tpl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints_mm1/ARCHIVE/stage2b_best.pt")
    p.add_argument("--tag", default="mm1")
    p.add_argument("--mhc-source", choices=["crystal", "template"], default="crystal")
    p.add_argument("--out-dir", default=str(HERE / "predictions"))
    p.add_argument("--csv", default=None)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--stage", type=int, default=2)
    p.add_argument("--angle-input", choices=["layernorm", "raw"], default="layernorm")
    p.add_argument("--leak-test", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: first N complexes")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bm = pd.read_csv(HERE / "pRMSD_benchmark.tsv", sep="\t")
    if args.limit:
        bm = bm.head(args.limit)
    conv = HERE / "converted_pdbs"
    out_dir = Path(args.out_dir) / f"{args.tag}_{args.mhc_source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames="identity",
                           angle_input=args.angle_input)
    net.set_stage(args.stage)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["trainable"], strict=False)
    net.eval()
    print(f"loaded {args.ckpt}  gstep={ck.get('global_step')}  stage={ck.get('stage')}")

    templates = pick_templates(bm) if args.mhc_source == "template" else {}
    ca_i = rc.atom_order["CA"]
    bb_idx = [rc.atom_order[a] for a in ("N", "CA", "C")]

    # warm-up so the first complex is not charged for CUDA/cuDNN initialisation
    _r = bm.iloc[0]
    _ex = m1.parse_example(conv / f"{_r.PDB}.pdb", _r.peptide, _r.mhc_seq,
                           anchor_sets(len(_r.peptide))[0], 1, return_backbone=True)
    _ex["teacher_plddt"] = torch.zeros(_ex["aatype"].shape[0])
    _ex["teacher_pae"] = torch.zeros(_ex["aatype"].shape[0], _ex["aatype"].shape[0])
    _b = m1.move_batch(m1.collate_with_teacher([_ex]), dev)
    for _ in range(3):
        net.predict(_b)
    if dev == "cuda":
        torch.cuda.synchronize()

    rows, per_anchor = [], []
    t_start = time.perf_counter()
    for n_done, (_, r) in enumerate(bm.iterrows(), 1):
        pdb_path = conv / f"{r.PDB}.pdb"
        L = len(r.peptide)
        ancs = anchor_sets(L)

        # ---- reference: the crystal structure, read from the SAME converted file
        ref = ofp.from_pdb_string(pdb_path.read_text())
        ref_xyz, ref_mask = ref.atom_positions, ref.atom_mask
        n_tot = ref_xyz.shape[0]
        n_mhc, n_pep = len(r.mhc_seq), L
        pep = np.zeros(n_tot, bool); pep[n_mhc:] = True
        mhc = ~pep

        exs = []
        for a in ancs:
            ex = m1.parse_example(pdb_path, r.peptide, r.mhc_seq, a, 1,
                                  return_backbone=True)
            ex["teacher_plddt"] = torch.zeros(n_tot)
            ex["teacher_pae"] = torch.zeros(n_tot, n_tot)
            exs.append(ex)

        if args.mhc_source == "template":
            tpl_id, tpl_ident = templates[r.PDB]
            trow = bm[bm.PDB == tpl_id].iloc[0]
            tref = ofp.from_pdb_string((conv / f"{tpl_id}.pdb").read_text())
            t_mhc = tref.atom_positions[:len(trow.mhc_seq)]          # [M,37,3]
            # align the template groove onto this target's groove, then donate it
            k = min(len(trow.mhc_seq), n_mhc)
            rot, mu_m, mu_r = kabsch(t_mhc[:k, ca_i], ref_xyz[:k][:, ca_i])
            t_al = (t_mhc - mu_m) @ rot.T + mu_r
            donor = np.zeros((n_mhc, 3, 3), np.float32)
            donor[:k] = t_al[:k][:, bb_idx]
            donor[k:] = ref_xyz[k:n_mhc][:, bb_idx]                  # pad with own tail
            for ex in exs:
                ex["teacher_bb"][:n_mhc] = torch.from_numpy(donor)
        else:
            tpl_id, tpl_ident = "", float("nan")

        b = m1.move_batch(m1.collate_with_teacher(exs), dev)
        t0 = time.perf_counter()
        out = net.predict(b)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_batch = time.perf_counter() - t0

        masks = make_atom14_masks({"aatype": b["aatype"]})
        atom37 = atom14_to_atom37(out["atom14"], masks).float().cpu().numpy()  # [K,N,37,3]
        pred_mask = masks["atom37_atom_exists"].cpu().numpy()                  # [K,N,37]
        plddt = m1.plddt_from_logits(out["plddt_logits"],
                                     getattr(net, "plddt_no_bins", 50)).float().cpu().numpy()

        cand = []
        for k, a in enumerate(ancs):
            rot, mu_m, mu_r = kabsch(atom37[k][mhc, ca_i], ref_xyz[mhc][:, ca_i])
            sup = (atom37[k] - mu_m) @ rot.T + mu_r
            both = (pred_mask[k] * ref_mask).astype(bool)

            def sel(res_m, atoms=None):
                m = np.zeros_like(both); m[res_m] = both[res_m]
                if atoms is not None:
                    keep = np.zeros(37, bool); keep[atoms] = True
                    m &= keep[None, :]
                return sup[m], ref_xyz[m]

            cand.append(dict(
                PDB=r.PDB, anchors=a, peptide=r.peptide, pep_len=L,
                pep_plddt=float(plddt[k][pep].mean()),
                complex_plddt=float(plddt[k].mean()),
                pep_ca_rmsd=rmsd(*sel(pep, [ca_i])),
                pep_bb_rmsd=rmsd(*sel(pep, bb_idx)),
                pep_allatom_rmsd=rmsd(*sel(pep)),
                mhc_ca_rmsd=rmsd(*sel(mhc, [ca_i])),
                _sup=sup, _pm=pred_mask[k], _pl=plddt[k]))

        per_anchor += [{k: v for k, v in c.items() if not k.startswith("_")}
                       for c in cand]
        best = max(cand, key=lambda c: c["pep_plddt"])

        leak = float("nan")
        if args.leak_test:
            kbest = ancs.index(best["anchors"])
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            pep_t = torch.from_numpy(pep).to(dev)
            b2["teacher_bb"][:, pep_t] += torch.tensor([10.0, 0.0, 0.0], device=dev)
            for key in ("teacher_atom14", "teacher_chi", "teacher_ca",
                        "teacher_plddt", "teacher_pae"):
                if key in b2 and torch.is_tensor(b2[key]):
                    b2[key] = torch.randn_like(b2[key]) * 10.0
            d = (net.predict(b2)["atom14"] - out["atom14"])[kbest][pep_t]
            leak = float(d.abs().max()) if d.numel() else 0.0

        rec = {k: v for k, v in best.items() if not k.startswith("_")}
        rec.update(n_anchor_variants=len(ancs), template_pdb=tpl_id,
                   template_mhc_identity=tpl_ident, leak_delta_A=leak,
                   t_all_anchors_ms=t_batch * 1e3,
                   t_per_structure_ms=t_batch * 1e3 / len(ancs),
                   Allele=str(r.Allele).strip(), Resolution=r.Resolution)
        rows.append(rec)

        bf = np.repeat(best["_pl"][:, None], 37, axis=1) * best["_pm"]
        prot = ofp.Protein(atom_positions=best["_sup"],
                           aatype=b["aatype"][0].cpu().numpy(),
                           atom_mask=best["_pm"],
                           residue_index=np.arange(1, n_tot + 1),
                           b_factors=bf, chain_index=np.where(pep, 1, 0))
        (out_dir / f"{r.PDB}_pred.pdb").write_text(ofp.to_pdb(prot))

        if n_done % 25 == 0 or n_done == len(bm):
            el = time.perf_counter() - t_start
            print(f"  [{n_done}/{len(bm)}] {el:6.1f}s  "
                  f"running mean pep_ca={np.mean([x['pep_ca_rmsd'] for x in rows]):.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    csv = Path(args.csv) if args.csv else HERE / f"results_{args.tag}_{args.mhc_source}.csv"
    df.to_csv(csv, index=False)
    pd.DataFrame(per_anchor).to_csv(
        csv.with_name(csv.stem + "_per_anchor.csv"), index=False)

    print(f"\n=== {args.tag}  mhc_source={args.mhc_source}  N={len(df)} ===")
    for c in ("pep_ca_rmsd", "pep_bb_rmsd", "pep_allatom_rmsd", "mhc_ca_rmsd"):
        v = df[c]
        print(f"  {c:18s} mean {v.mean():6.3f}  median {v.median():6.3f}  "
              f"<1A {100*(v<1).mean():5.1f}%  <2A {100*(v<2).mean():5.1f}%")
    print(f"  mean peptide pLDDT of the kept variant: {df.pep_plddt.mean():.1f}")
    print(f"  {df.t_per_structure_ms.median():.1f} ms/structure "
          f"(batched over the {int(df.n_anchor_variants.median())} anchor variants)")
    if args.leak_test:
        mx = float(df.leak_delta_A.max())
        print(f"  LEAK max |delta peptide atom| = {mx:.3e} A  "
              f"{'PASS' if mx == 0.0 else '*** FAIL ***'}")
    print(f"\nwrote {csv}\nPDBs -> {out_dir}")


if __name__ == "__main__":
    main()
