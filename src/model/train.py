"""
PMGen-v2 distillation training entrypoint (thin CLI; heavy lifting in utils.py).

Trains ONLY the small encoder; the AF2 structure module + pLDDT/PAE heads stay
frozen. Loss = λ_fape·FAPE + λ_plddt·CE(pLDDT) + λ_pae·CE(PAE).

Examples
--------
  # local smoke / overfit on the 15-example dummy set
  ~/miniforge3/envs/pmgen2/bin/python src/model/train.py \\
      --dummy --variant 7 --epochs 20 --bs 3 --lr 3e-3

  # real run (needs teacher PMGen outputs under --af-root, one dir per anchor id)
  ... train.py --scheme two_axis --fold 1 --variant 7 --af-root <dir> --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", type=int, default=7, choices=range(1, 8))
    p.add_argument("--scheme", default="two_axis", choices=["two_axis", "hla_only"])
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--dummy", action="store_true",
                   help="train on data/test/ (15 class-I examples) for local runs")
    p.add_argument("--af-root", default=None,
                   help="real mode (on-the-fly PDB read): dir with one PMGen "
                        "teacher output sub-dir per anchor id")
    p.add_argument("--h5-dir", default=None,
                   help="preprocessed HDF5 store dir (fastest; from preprocess.py)")
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="dir for checkpoints (<ckpt-dir>/<run-name>/{last,best}.pt)")
    p.add_argument("--run-name", default=None,
                   help="checkpoint subdir (default variant{v}_{scheme}_fold{f})")
    p.add_argument("--resume", default=None,
                   help="path to a last.pt to resume a preempted run")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambdas", type=float, nargs=3, default=(1.0, 0.01, 0.01),
                   metavar=("FAPE", "PLDDT", "PAE"),
                   help="loss weights. Default 1/0.01/0.01 (AF2-style: geometry "
                        "dominates; pLDDT/PAE are light auxiliary heads)")
    p.add_argument("--peptide-weight", type=float, default=1.0,
                   help="up-weight peptide residues/pairs in the loss "
                        "(1.0=uniform/unchanged; ~3 emphasizes the peptide)")
    p.add_argument("--recycles", type=int, default=0,
                   help="recycle iterations (0=off). Eval uses this many; training "
                        "samples 1..N per step. Changes the architecture (adds a "
                        "recycling embedder) -> use a fresh run dir.")
    p.add_argument("--recycle-probs", type=float, nargs="+", default=None,
                   metavar="P", help="per-step recycle distribution P(nr=0,1,2,...) "
                        "during training, e.g. 0.8 0.2 = 80%% no recycle / 20%% one. "
                        "Overrides the default uniform 1..recycles sampling.")
    p.add_argument("--eval-recycles", type=int, default=None,
                   help="recycle count to use at validation (default: model max)")
    p.add_argument("--unfreeze-sm", type=float, default=0.0,
                   help="%% of the structure module to unfreeze (last params first)")
    p.add_argument("--unfreeze-plddt", type=float, default=0.0,
                   help="%% of the pLDDT head to unfreeze")
    p.add_argument("--unfreeze-pae", type=float, default=0.0,
                   help="%% of the PAE/TM head to unfreeze")
    p.add_argument("--anchor-relpos", action="store_true",
                   help="re-number the peptide from its anchors so the alignment "
                        "register reaches the SM via relpos (experimental A/B; "
                        "new architecture-equivalent -> use a fresh run dir)")
    p.add_argument("--plddt-weight-struct", action="store_true",
                   help="weight each example's FAPE by the teacher's median peptide "
                        "pLDDT/100, so low-confidence teacher structures contribute "
                        "less to the geometry loss (loss-only; resume-safe)")
    p.add_argument("--plddt-weight-floor", type=float, default=0.1,
                   help="lower bound on the per-example pLDDT weight (keeps the "
                        "weakest examples from vanishing entirely)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100,
                   help="log all loss terms + lr + it/s every N steps (0=off)")
    p.add_argument("--ckpt-every", type=int, default=2000,
                   help="write a resumable last.pt every N steps (0=epoch-only)")
    p.add_argument("--no-tensorboard", action="store_true",
                   help="disable the TensorBoard writer (metrics.csv still written)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    # Stream logs live: stdout is block-buffered when redirected to a file
    # (e.g. SLURM --output), so flush on every newline instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)

    # Cheap skip for already-finished runs: peek the (small) checkpoint and exit
    # before building the dataset / loading the frozen stack. Re-runs if you later
    # raise --epochs (then saved epoch < requested). Corrupt peek -> fall through.
    if args.resume and Path(args.resume).exists():
        try:
            import torch
            done = int(torch.load(args.resume, map_location="cpu",
                                  weights_only=False).get("epoch", 0))
            if done >= args.epochs:
                print(f"[train] {args.run_name or args.resume}: already completed "
                      f"{done}/{args.epochs} epochs — nothing to do.", flush=True)
                return
        except Exception as e:
            print(f"[train] could not peek {args.resume} ({e}); continuing.",
                  flush=True)

    U.run_training(
        variant=args.variant, scheme=args.scheme, fold=args.fold,
        dummy=args.dummy, af_root=args.af_root, h5_dir=args.h5_dir,
        ckpt_dir=args.ckpt_dir, run_name=args.run_name, resume=args.resume,
        epochs=args.epochs, bs=args.bs, lr=args.lr, lambdas=tuple(args.lambdas),
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, amp=args.amp,
        num_workers=args.num_workers, seed=args.seed, device=args.device,
        log_every=args.log_every, ckpt_every=args.ckpt_every,
        tensorboard=not args.no_tensorboard, peptide_weight=args.peptide_weight,
        recycles=args.recycles, recycle_probs=args.recycle_probs,
        eval_recycles=args.eval_recycles, unfreeze_sm=args.unfreeze_sm,
        unfreeze_plddt=args.unfreeze_plddt, unfreeze_pae=args.unfreeze_pae,
        anchor_relpos=args.anchor_relpos,
        plddt_weight_struct=args.plddt_weight_struct,
        plddt_weight_floor=args.plddt_weight_floor,
    )


if __name__ == "__main__":
    main()
