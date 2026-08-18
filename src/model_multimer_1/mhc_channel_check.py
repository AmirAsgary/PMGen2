"""Does PEPTIDE information reach the model through the INPUT MHC BACKBONE?

`_leak_check` proves the peptide's own `teacher_bb` cannot reach the trunk under
`--pep-frames identity`. It does NOT cover a subtler channel:

    the MHC backbone we feed in was predicted by AlphaFold WITH THE PEPTIDE PRESENT,
    so its induced-fit groove conformation is peptide-specific.

That is not cheating — handing the model an MHC structure is the design. It is a
TRAIN/DEPLOY MISMATCH risk: at deployment the MHC cannot come from a co-fold with the
true peptide (you do not have it yet), and val here uses co-folded MHC too, so both
train and val numbers would be inflated together and the gap would never show up.

Two tests, both on REAL structures:

  A. SIBLING SWAP — siblings share a sequence but differ in anchor, so they have
     different peptide poses and slightly different grooves. Predict with sibling A's
     anchor but sibling B's MHC. If the prediction drifts TOWARD B's peptide, the model
     is reading pose information out of the groove.

  B. CROSS-PEPTIDE SWAP (the deployment case) — same MHC allele, DIFFERENT peptide.
     Predict peptide P1 using an MHC backbone co-folded with an unrelated peptide P2.
     This is what inference will actually look like. If accuracy collapses here, the
     model depends on a co-folded groove and will not transfer.

    $PY src/model_multimer_1/mhc_channel_check.py --ckpt <ckpt>
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import model as MM                                                    # noqa: E402


def _rmsd(mob, ref, sel, tgt):
    mc, rc = mob[sel].mean(0), ref[sel].mean(0)
    P, Q = mob[sel] - mc, ref[sel] - rc
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((((mob - mc) @ R.T)[tgt] - (ref - rc)[tgt]) ** 2).sum(1).mean()))


def _mk(g, dev, bb=None, anchor=None):
    aat = np.array(g["aatype"])
    b = {"aatype": torch.as_tensor(aat, dtype=torch.long, device=dev)[None],
         "segment_id": torch.as_tensor(np.array(g["segment_id"]), dtype=torch.long, device=dev)[None],
         "anchor": torch.as_tensor(np.array(g["anchor"]) if anchor is None else anchor,
                                   dtype=torch.float32, device=dev)[None],
         "teacher_bb": torch.as_tensor(np.array(g["teacher_bb"]) if bb is None else bb,
                                       dtype=torch.float32, device=dev)[None]}
    b["seq_mask"] = torch.ones_like(b["aatype"], dtype=torch.float32)
    return b


def _pred(net, b, dev):
    with torch.no_grad():
        if dev == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return net.predict(b)["ca"][0].float().cpu().numpy().astype(np.float64)
        return net.predict(b)["ca"][0].float().cpu().numpy().astype(np.float64)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames="identity")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("trainable", ck.get("model", ck))
    if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in sd.values()):
        raise SystemExit("FATAL: checkpoint contains non-finite tensors.")
    net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"ckpt {args.ckpt} (epoch {ck.get('epoch','?')}, gstep {ck.get('global_step','?')})")

    shards = sorted(glob.glob(f"{args.h5_dir}/*.index.csv"))
    random.Random(args.seed).shuffle(shards)
    A, B = [], []
    for f in shards:
        if len(A) >= args.n and len(B) >= args.n:
            break
        base = f.replace(".index.csv", "")
        idx = pd.read_csv(f)
        try:
            h = h5py.File(base + ".h5", "r")
        except Exception:
            continue
        # ---- A: sibling swap ----
        for bid, g in idx.groupby("base_id"):
            if len(A) >= args.n:
                break
            ids = [i for i in g.id.tolist() if i in h]
            if len(ids) < 2:
                continue
            npep = int(g.n_pep.iloc[0])
            gA, gB = h[ids[0]], h[ids[1]]
            if np.array(gA["aatype"]).shape != np.array(gB["aatype"]).shape:
                continue
            n = np.array(gA["aatype"]).shape[0]
            mhc, pep = np.arange(0, n - npep), np.arange(n - npep, n)
            bbA, bbB = np.array(gA["teacher_bb"]), np.array(gB["teacher_bb"])
            tA, tB = bbA[:, 1, :].astype(np.float64), bbB[:, 1, :].astype(np.float64)
            # keep A's peptide bb slot irrelevant (identity frames); swap only MHC rows
            bb_mix = bbA.copy()
            bb_mix[mhc] = bbB[mhc]
            pAA = _pred(net, _mk(gA, dev), dev)
            pBA = _pred(net, _mk(gA, dev, bb=bb_mix), dev)
            A.append({"mhc_input_delta": _rmsd(bbA[:, 1, :].astype(np.float64),
                                               bbB[:, 1, :].astype(np.float64), mhc, mhc),
                      "pred_shift": _rmsd(pAA, pBA, mhc, pep),
                      "toB_before": _rmsd(pAA, tB, mhc, pep),
                      "toB_after": _rmsd(pBA, tB, mhc, pep),
                      "toA_before": _rmsd(pAA, tA, mhc, pep),
                      "toA_after": _rmsd(pBA, tA, mhc, pep)})
        # ---- B: cross-peptide swap (same allele, different peptide) ----
        idx["allele"] = idx.id.str.split("_").str[0]
        for al, g in idx.groupby("allele"):
            if len(B) >= args.n:
                break
            bys = {}
            for _, r in g.iterrows():
                if r.id in h:
                    bys.setdefault((r.n_mhc, r.n_pep), []).append(r.id)
            for (nm, npep), lst in bys.items():
                if len(B) >= args.n:
                    break
                bases = {}
                for i in lst:
                    bases.setdefault(i.rsplit("_", 1)[0], i)
                if len(bases) < 2:
                    continue
                (b1, i1), (b2, i2) = list(bases.items())[:2]
                g1, g2 = h[i1], h[i2]
                a1, a2 = np.array(g1["aatype"]), np.array(g2["aatype"])
                if a1.shape != a2.shape:
                    continue
                n = a1.shape[0]
                mhc, pep = np.arange(0, n - npep), np.arange(n - npep, n)
                if not np.array_equal(a1[mhc], a2[mhc]):
                    continue                       # need the SAME MHC sequence
                bb1, bb2 = np.array(g1["teacher_bb"]), np.array(g2["teacher_bb"])
                t1 = bb1[:, 1, :].astype(np.float64)
                mix = bb1.copy()
                mix[mhc] = bb2[mhc]                # foreign (but same-allele) groove
                own = _pred(net, _mk(g1, dev), dev)
                for_ = _pred(net, _mk(g1, dev, bb=mix), dev)
                B.append({"mhc_input_delta": _rmsd(t1, bb2[:, 1, :].astype(np.float64), mhc, mhc),
                          "acc_own_mhc": _rmsd(own, t1, mhc, pep),
                          "acc_foreign_mhc": _rmsd(for_, t1, mhc, pep)})
        h.close()

    da, db = pd.DataFrame(A), pd.DataFrame(B)
    print("\n=========== TEST A: sibling MHC swap (does the groove carry the pose?) ===========")
    if len(da):
        print(f"  n={len(da)}   MHC input changed by {da.mhc_input_delta.median():.2f} A (median)")
        print(f"  prediction moved                : {da.pred_shift.median():6.2f} A")
        print(f"  distance to sibling B's peptide : {da.toB_before.median():6.2f} -> "
              f"{da.toB_after.median():6.2f} A")
        print(f"  distance to sibling A's peptide : {da.toA_before.median():6.2f} -> "
              f"{da.toA_after.median():6.2f} A")
        drift = da.toB_before.median() - da.toB_after.median()
        print(f"  DRIFT TOWARD B                  : {drift:+6.2f} A")
        if drift > 0.25:
            print("  => the input groove DOES carry peptide-pose information the model uses.")
        else:
            print("  => no meaningful drift: the groove is used as a scaffold, not a pose hint.")
    print("\n=========== TEST B: cross-peptide MHC swap (the DEPLOYMENT case) ===========")
    if len(db):
        print(f"  n={len(db)}   foreign MHC differs by {db.mhc_input_delta.median():.2f} A (median)")
        print(f"  accuracy with its OWN co-folded MHC : {db.acc_own_mhc.median():6.2f} A")
        print(f"  accuracy with a FOREIGN same-allele  : {db.acc_foreign_mhc.median():6.2f} A")
        pen = db.acc_foreign_mhc.median() - db.acc_own_mhc.median()
        print(f"  DEPLOYMENT PENALTY                   : {pen:+6.2f} A")
        if pen > 0.5:
            print("  => the model LEANS on the co-folded groove; expect this much degradation")
            print("     at inference, where no co-folded MHC exists. Reported val is optimistic.")
        else:
            print("  => robust to the groove's provenance: val numbers should transfer.")
    out = Path("outputs/mm1_anchor"); out.mkdir(parents=True, exist_ok=True)
    da.to_csv(out / "mhc_sibling_swap.csv", index=False)
    db.to_csv(out / "mhc_cross_swap.csv", index=False)
    print(f"\nwrote {out}/mhc_*.csv")


if __name__ == "__main__":
    main()
