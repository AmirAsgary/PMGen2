"""
pLDDT prediction quality on the validation set, for model_1 or model_2.

Collects per-residue (predicted pLDDT, teacher pLDDT) pairs and visualises how well
the confidence head is calibrated — overall and, more importantly, on the PEPTIDE
(the MHC pLDDT is uniformly high and easy; the peptide is where confidence matters).

Predicted pLDDT is taken the same way each model produces it:
  * model_1: the frozen pLDDT head on the encoder->SM single rep (model forward).
  * model_2: the aux pLDDT head on the clean (teacher) structure embedding — i.e.
    exactly the quantity its training loss supervises (no sampling needed).

Outputs <out-dir>/plddt_eval.png (+ .csv summary). Run ON THE CLUSTER (GPU + store):
  python src/post_structure_prediction_processing/eval_plddt.py --model 1 \
      --ckpt checkpoints/two_axis_fold1_variant7_pw5.0_rc3/last.pt \
      --h5-dir data/processed/h5_store --out-dir outputs/plddt/model1
  python src/post_structure_prediction_processing/eval_plddt.py --model 2 \
      --ckpt checkpoints_model2/mhcdiff_two_axis_fold1/last.pt \
      --h5-dir data/processed/h5_store_sc --out-dir outputs/plddt/model2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]


def _spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    d = (ar.std() * br.std())
    return float((ar * br).mean() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
def collect_model1(args, device):
    sys.path.insert(0, str(_SRC / "model"))
    import utils as U
    cfg = json.loads((Path(args.ckpt).parent / "config.json").read_text())
    model = U.DistillModel(
        cfg["variant"], device=device, recycles=cfg.get("recycles", 0),
        unfreeze_sm=cfg.get("unfreeze_sm", 0.0),
        unfreeze_plddt=cfg.get("unfreeze_plddt", 0.0),
        unfreeze_pae=cfg.get("unfreeze_pae", 0.0),
        anchor_relpos=cfg.get("anchor_relpos", False)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ck["encoder"])
    if ck.get("frozen_trainable"):
        model.load_state_dict(ck["frozen_trainable"], strict=False)
    model.eval()

    ds = U.build_h5_dataset(args.h5_dir, args.scheme, args.fold, args.split)
    loader = U.make_dataloader(ds, args.bs, shuffle=False, num_workers=args.num_workers)
    er = cfg.get("recycles", 0) or None
    pred, teach, isp, seen = [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            batch = U.move_batch(batch, device)
            _, plddt_logits, _ = model(batch, return_frames=False, num_recycles=er)
            pl = U.plddt_from_logits(plddt_logits, plddt_logits.shape[-1])   # [B,N]
            sm = batch["seq_mask"].bool()
            pep = U.peptide_mask_from_batch(batch["seq_mask"],
                                            batch["segment_id"]).bool()
            for b in range(pl.shape[0]):
                m = sm[b]
                pred.append(pl[b][m].cpu().numpy())
                teach.append(batch["teacher_plddt"][b][m].cpu().numpy())
                isp.append(pep[b][m].cpu().numpy())
            seen += pl.shape[0]
            if args.max_graphs and seen >= args.max_graphs:
                break
    return (np.concatenate(pred), np.concatenate(teach),
            np.concatenate(isp).astype(bool))


def collect_model2(args, device):
    sys.path.insert(0, str(_SRC / "model_2"))
    import model as M
    import pyg_data as PD
    from torch_geometric.loader import DataLoader
    cfg = json.loads((Path(args.ckpt).parent / "config.json").read_text())
    net = M.MHCDiff(T=cfg["timesteps"], mhc_scale=cfg["mhc_scale"],
                    h_dim=cfg["hidden"], n_layers=cfg["layers"], k=cfg["k"],
                    use_cross=not cfg["no_cross"], device=device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device,
                                   weights_only=False)["model"])
    net.eval()
    base = M.PD.m1.build_h5_dataset(args.h5_dir, args.scheme, args.fold, args.split)
    val = PD.H5GraphDataset(base)
    pred, teach, isp, seen = [], [], [], 0
    with torch.no_grad():
        for data in DataLoader(val, batch_size=args.bs):
            data = data.to(device)
            # predict pLDDT from the CLEAN (teacher) structure — the quantity the
            # training loss supervises (h_clean = aux_embed(x0); x0 = teacher coords).
            x0 = M.center(data.pos / M.COORD_SCALE, data.batch)
            h = net._aux_embed(x0, data)
            pl = (net.plddt_head(h).squeeze(-1).sigmoid() * 100.0).cpu().numpy()
            pred.append(pl)
            teach.append(data.teacher_plddt.cpu().numpy())
            isp.append(data.pep.cpu().numpy().astype(bool))
            seen += data.num_graphs
            if args.max_graphs and seen >= args.max_graphs:
                break
    return np.concatenate(pred), np.concatenate(teach), np.concatenate(isp)


# --------------------------------------------------------------------------- #
def _stats(pred, teach):
    r = float(np.corrcoef(pred, teach)[0, 1]) if len(pred) > 2 else float("nan")
    return {"n": len(pred), "pearson": r, "spearman": _spearman(pred, teach),
            "mae": float(np.abs(pred - teach).mean())}


def report_plot(pred, teach, isp, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pep, mhc = isp, ~isp
    s_all, s_pep, s_mhc = _stats(pred, teach), _stats(pred[pep], teach[pep]), \
        _stats(pred[mhc], teach[mhc])
    print(f"\n=== {tag} pLDDT prediction (val, n={len(pred):,}) ===")
    for name, s in [("ALL", s_all), ("PEPTIDE", s_pep), ("MHC", s_mhc)]:
        print(f"  {name:<8} n={s['n']:>7,}  Pearson {s['pearson']:+.3f}  "
              f"Spearman {s['spearman']:+.3f}  MAE {s['mae']:.2f}")
    import pandas as pd
    pd.DataFrame([{"subset": k, **v} for k, v in
                  [("all", s_all), ("peptide", s_pep), ("mhc", s_mhc)]]
                 ).to_csv(out_dir / "plddt_eval.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    for axi, mask, name, s in [(ax[0, 0], slice(None), "all residues", s_all),
                               (ax[0, 1], pep, "PEPTIDE only", s_pep)]:
        x, y = teach[mask], pred[mask]
        hb = axi.hexbin(x, y, gridsize=45, cmap="viridis", mincnt=1,
                        extent=(0, 100, 0, 100), bins="log")
        axi.plot([0, 100], [0, 100], "r--", lw=1, label="y=x")
        axi.set(xlim=(0, 100), ylim=(0, 100), xlabel="teacher pLDDT",
                ylabel="predicted pLDDT",
                title=f"{name}  (Pearson {s['pearson']:+.2f}, MAE {s['mae']:.1f})")
        axi.legend(fontsize=8, loc="upper left")
        fig.colorbar(hb, ax=axi, label="log count")

    # calibration: mean predicted within teacher bins (peptide vs MHC)
    bins = np.linspace(0, 100, 11)
    ctr = (bins[:-1] + bins[1:]) / 2
    for mask, lbl, col in [(pep, "peptide", "#C44E52"), (mhc, "MHC", "#4C72B0")]:
        idx = np.digitize(teach[mask], bins) - 1
        mu = [pred[mask][idx == k].mean() if (idx == k).any() else np.nan
              for k in range(10)]
        sd = [pred[mask][idx == k].std() if (idx == k).any() else np.nan
              for k in range(10)]
        ax[1, 0].errorbar(ctr, mu, yerr=sd, marker="o", color=col, label=lbl, capsize=3)
    ax[1, 0].plot([0, 100], [0, 100], "k--", lw=1, label="perfect")
    ax[1, 0].set(xlim=(0, 100), ylim=(0, 100), xlabel="teacher pLDDT bin",
                 ylabel="mean predicted pLDDT", title="calibration")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.25)

    # distributions
    ax[1, 1].hist(teach[pep], bins=40, range=(0, 100), density=True, alpha=0.5,
                  color="#55A868", label="teacher (pep)")
    ax[1, 1].hist(pred[pep], bins=40, range=(0, 100), density=True, alpha=0.5,
                  color="#C44E52", label="predicted (pep)")
    ax[1, 1].set(xlabel="pLDDT", ylabel="density",
                 title="peptide pLDDT distribution")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"{tag} — pLDDT prediction on validation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "plddt_eval.png", dpi=140)
    plt.close(fig)
    print(f"\n[plot] wrote {out_dir/'plddt_eval.png'}")
    print(f"[csv]  wrote {out_dir/'plddt_eval.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=int, choices=[1, 2], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--scheme", default="two_axis")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-graphs", type=int, default=2000,
                    help="cap examples for speed (0 = all val)")
    ap.add_argument("--out-dir", default="outputs/plddt")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == 1:
        pred, teach, isp = collect_model1(args, device)
        tag = f"model_1 ({Path(args.ckpt).parent.name})"
    else:
        pred, teach, isp = collect_model2(args, device)
        tag = f"model_2 ({Path(args.ckpt).parent.name})"
    report_plot(pred, teach, isp, Path(args.out_dir), tag)


if __name__ == "__main__":
    main()
