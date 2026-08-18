"""Does the model actually CONDITION ON THE ANCHOR? — the core premise of PMGen.

The store holds, for each pMHC complex, several AlphaFold structures that differ ONLY
in the anchor given to AlphaFold (verified: 100% of the 225,616 multi-structure
complexes have all-distinct anchors). So the anchor is not metadata — it is the input
that distinguishes otherwise identical training examples, and learning
anchor -> structure IS the task.

This measures whether that was learned, the same way `_leak_check` measures the leak:
hold everything fixed, change ONLY the anchor, and see how far the prediction moves.

    DATA response  : teacher peptide of sibling A vs sibling B (superposed on the MHC)
                     — how much AlphaFold moves the peptide for this anchor change.
    MODEL response : predict(x, anchor_A) vs predict(x, anchor_B), same MHC, same
                     sequence — how much OUR model moves it for the same change.

    ratio = MODEL / DATA
      ~1.0  the anchor conditioning was learned at the right magnitude
      ~0    the model IGNORES the anchor — it is predicting one average pose per
            sequence, and the method's premise has not been learned
      >>1   over-reacting to the anchor

Also reports accuracy against each sibling's own teacher, so a model that ignores the
anchor is visibly stuck near the midpoint of the sibling poses.

    $PY src/model_multimer_1/anchor_sensitivity.py --ckpt checkpoints_mm1/<run>/last.pt
"""
from __future__ import annotations

import argparse
import glob
import itertools
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import model as MM                                                    # noqa: E402


def _kabsch_rmsd(mob, ref, sel, tgt):
    mc, rc = mob[sel].mean(0), ref[sel].mean(0)
    P, Q = mob[sel] - mc, ref[sel] - rc
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((((mob - mc) @ R.T)[tgt] - (ref - rc)[tgt]) ** 2).sum(1).mean()))


def _batch(g, dev):
    b = {
        "aatype": torch.as_tensor(np.array(g["aatype"]), dtype=torch.long, device=dev)[None],
        "segment_id": torch.as_tensor(np.array(g["segment_id"]), dtype=torch.long, device=dev)[None],
        "anchor": torch.as_tensor(np.array(g["anchor"]), dtype=torch.float32, device=dev)[None],
        "teacher_bb": torch.as_tensor(np.array(g["teacher_bb"]), dtype=torch.float32, device=dev)[None],
    }
    b["seq_mask"] = torch.ones_like(b["aatype"], dtype=torch.float32)
    return b


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--h5-dir", default="data/processed/h5_store_sc")
    p.add_argument("--n", type=int, default=120, help="complexes to test")
    p.add_argument("--n-trunk", type=int, default=3)
    p.add_argument("--pep-frames", choices=["teacher", "identity"], default="identity")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MM.MultimerModel(n_trunk=args.n_trunk, device=dev, pep_frames=args.pep_frames)
    tag = "UNTRAINED (random init)"
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("trainable", ck.get("model", ck))
        bad = [k for k, v in sd.items()
               if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            raise SystemExit(f"FATAL: {len(bad)}/{len(sd)} tensors are non-finite — "
                             f"this checkpoint is dead (e.g. {bad[:2]}).")
        net.load_state_dict(sd, strict=False)
        tag = f"{args.ckpt} (epoch {ck.get('epoch','?')}, gstep {ck.get('global_step','?')})"
    net.eval()

    import pandas as pd
    shards = sorted(glob.glob(f"{args.h5_dir}/*.index.csv"))
    random.Random(args.seed).shuffle(shards)
    rows = []
    for f in shards:
        if len(rows) >= args.n:
            break
        base = f.replace(".index.csv", "")
        idx = pd.read_csv(f)
        try:
            h = h5py.File(base + ".h5", "r")
        except Exception:
            continue
        for bid, g in idx.groupby("base_id"):
            if len(rows) >= args.n:
                break
            ids = [i for i in g.id.tolist() if i in h]
            if len(ids) < 2:
                continue
            npep = int(g.n_pep.iloc[0])
            A, B = ids[0], ids[1]
            gA, gB = h[A], h[B]
            if np.array(gA["aatype"]).shape != np.array(gB["aatype"]).shape:
                continue
            ancA = np.array(gA["anchor"])
            ancB = np.array(gB["anchor"])
            if np.array_equal(ancA, ancB):
                continue                       # need a real anchor change
            n = ancA.shape[0]
            mhc = np.arange(0, n - npep)
            pep = np.arange(n - npep, n)
            tA = np.array(gA["teacher_bb"])[:, 1, :].astype(np.float64)
            tB = np.array(gB["teacher_bb"])[:, 1, :].astype(np.float64)
            data_resp = _kabsch_rmsd(tA, tB, mhc, pep)

            bA = _batch(gA, dev)
            bA2 = dict(bA)
            bA2["anchor"] = torch.as_tensor(ancB, dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                if dev == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pA = net.predict(bA)["ca"][0].float().cpu().numpy().astype(np.float64)
                        pB = net.predict(bA2)["ca"][0].float().cpu().numpy().astype(np.float64)
                else:
                    pA = net.predict(bA)["ca"][0].float().cpu().numpy().astype(np.float64)
                    pB = net.predict(bA2)["ca"][0].float().cpu().numpy().astype(np.float64)
            model_resp = _kabsch_rmsd(pA, pB, mhc, pep)
            rows.append({
                "base": bid,
                "data_response": data_resp,
                "model_response": model_resp,
                "acc_vs_A": _kabsch_rmsd(pA, tA, mhc, pep),
                "acc_vs_B_using_anchorB": _kabsch_rmsd(pB, tB, mhc, pep),
                "acc_vs_B_using_anchorA": _kabsch_rmsd(pA, tB, mhc, pep),
            })
        h.close()

    d = pd.DataFrame(rows)
    print(f"\ncheckpoint : {tag}")
    print(f"complexes  : {len(d)}  (each: two siblings differing ONLY in the anchor)\n")
    print(f"  DATA  anchor response (AlphaFold) : median {d.data_response.median():6.2f} A")
    print(f"  MODEL anchor response (ours)      : median {d.model_response.median():6.2f} A")
    ratio = d.model_response.median() / max(d.data_response.median(), 1e-6)
    print(f"  ratio MODEL/DATA                  : {ratio:6.2f}  (magnitude only)")

    # THE VERDICT COMES FROM THE CONTROLLED COMPARISON, NOT THE RATIO.
    # An UNTRAINED model scores ratio ~2.9 here: a random net is chaotic, so perturbing
    # ANY input moves its output a lot. Magnitude of response therefore cannot
    # distinguish "uses the anchor" from "is unstable". The honest test is whether the
    # RIGHT anchor predicts sibling B better than the WRONG one, with the MHC, the
    # sequence and the target all held fixed — only the anchor differs.
    right = d.acc_vs_B_using_anchorB.median()
    wrong = d.acc_vs_B_using_anchorA.median()
    gain = wrong - right
    print(f"\n  accuracy vs sibling A (its own anchor)      : {d.acc_vs_A.median():6.2f} A")
    print(f"  accuracy vs sibling B using B's anchor      : {right:6.2f} A   <- right anchor")
    print(f"  accuracy vs sibling B using the WRONG anchor: {wrong:6.2f} A   <- wrong anchor")
    print(f"  ANCHOR GAIN (wrong - right)                 : {gain:+6.2f} A")
    frac = (d.acc_vs_B_using_anchorA > d.acc_vs_B_using_anchorB).mean()
    print(f"  right anchor wins on {frac*100:.0f}% of complexes")
    if gain <= 0.05:
        print("  => VERDICT: the anchor is NOT being used. The model predicts one pose")
        print("     per sequence; the conditioning that defines the method is not learned.")
    elif ratio < 0.6:
        print("  => VERDICT: the anchor IS used, but the model UNDER-responds")
        print(f"     ({ratio:.2f}x AlphaFold's own displacement) — conditioning is too weak.")
    else:
        print("  => VERDICT: the anchor is used, at a comparable magnitude.")
    out = Path("outputs/mm1_anchor"); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "anchor_sensitivity.csv", index=False)
    print(f"\nwrote {out/'anchor_sensitivity.csv'}")


if __name__ == "__main__":
    main()
