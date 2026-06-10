"""
GATE 5 — HDF5 preprocessing + streamable Dataset.

Builds a fake chunk layout from the 15 dummy examples (chunk.tsv + an `output`
symlink to data/test/pdbs), runs preprocess -> shard + index, reads examples back
from the H5 store and checks they round-trip equal to the direct parser, then runs
a short training entirely off the H5 store.

Run:  ~/miniforge3/envs/pmgen2/bin/python src/model/h5_test.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_COLS = ["peptide", "mhc_seq", "mhc_type", "anchors", "id",
              "hla_cluster_id", "peptide_cluster_id", "key_id"]


def make_fake_chunks(tmp: Path) -> Path:
    """One chunk from data/test/: chunk.tsv + output -> data/test/pdbs."""
    chunks_dir = tmp / "chunks"
    cdir = chunks_dir / "chunk_1"
    cdir.mkdir(parents=True)
    rows = U.dummy_rows()                      # has alphafold_output_path too
    with open(cdir / "chunk.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CHUNK_COLS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({"peptide": r["peptide"], "mhc_seq": r["mhc_seq"],
                        "mhc_type": r["mhc_type"], "anchors": r["anchors"],
                        "id": r["id"], "hla_cluster_id": "hc",
                        "peptide_cluster_id": "pc", "key_id": r["id"]})
    # output symlink -> the dir that CONTAINS alphafold/<id>/
    (cdir / "output").symlink_to((U.DUMMY_DIR / "pdbs").resolve())
    return chunks_dir


def check_roundtrip(h5_dir: Path) -> None:
    idx = U.load_h5_index(h5_dir)
    print(f"=== index: {len(idx)} examples, {idx['shard'].nunique()} shard(s), "
          f"{idx['base_id'].nunique()} base ids ===")
    assert len(idx) == len(U.dummy_rows())
    assert set(idx["base_id"]) == {b.rsplit("_", 1)[0] for b in idx["id"]}

    ds = U.build_h5_dataset(h5_dir, all_ids=True)
    rows = {r["id"]: r for r in U.dummy_rows()}
    for aid in [ds.ids[0], ds.ids[7], ds.ids[-1]]:
        h5ex = ds[ds.ids.index(aid)]
        r = rows[aid]
        ref = U.parse_example(
            Path(r["alphafold_output_path"]) /
            f"{aid}_model_1_model_2_ptm.pdb", r["peptide"], r["mhc_seq"],
            r["anchors"], r["mhc_type"], return_backbone=True)
        for k in ("aatype", "residue_index", "anchor", "segment_id"):
            assert torch.equal(h5ex[k], ref[k].to(h5ex[k].dtype)), f"{aid}:{k}"
        assert torch.allclose(h5ex["teacher_ca"], ref["teacher_ca"]), f"{aid}:ca"
        assert torch.allclose(h5ex["teacher_bb"], ref["teacher_bb"]), f"{aid}:bb"
        # pLDDT/PAE stored as float16 -> compare with f16 tolerance
        pl, pae = U.load_teacher_arrays(
            *U.find_teacher_files(Path(r["alphafold_output_path"]))[1:],
            int(ref["aatype"].shape[0]))
        assert torch.allclose(h5ex["teacher_plddt"], pl, atol=0.1), f"{aid}:plddt"
        assert torch.allclose(h5ex["teacher_pae"], pae, atol=0.05), f"{aid}:pae"
    print(f"  round-trip OK (3 ids): aatype/residue_index/anchor/segment exact, "
          f"ca/bb exact, plddt/pae within float16 tol.\n")


def check_collate_and_loss(h5_dir: Path) -> None:
    ds = U.build_h5_dataset(h5_dir, all_ids=True)
    batch = U.move_batch(U.collate_with_teacher([ds[0], ds[8]]), DEVICE)
    model = U.DistillModel(7, device=DEVICE).to(DEVICE)
    loss_mod = U.DistillLoss().to(DEVICE)
    ca, pl, pae, fr = model(batch, return_frames=True)
    total, terms = loss_mod(ca, pl, pae, fr, batch)
    print(f"=== H5 batch -> 3-term loss OK: total {float(total):.3f} "
          f"(fape {float(terms['fape']):.3f}) ===\n")
    assert torch.isfinite(total)


def check_training_from_h5(h5_dir: Path) -> None:
    print("=== short training from the H5 store (dummy=all ids, 8 epochs) ===")
    history, _ = U.run_training(variant=7, dummy=True, h5_dir=h5_dir, epochs=8,
                                bs=3, lr=3e-3, seed=0, device=DEVICE)
    first, last = history[0]["train"]["total"], history[-1]["train"]["total"]
    print(f"  train total {first:.3f} -> {last:.3f}")
    assert last < first, "training from H5 did not decrease the loss"
    print("  OK: training off the H5 store decreases the loss.\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        chunks_dir = make_fake_chunks(tmp)
        out_dir = tmp / "h5_store"
        U.preprocess_chunks(chunks_dir, out_dir, log=print)
        assert (out_dir / "chunk_1.h5").exists() and (out_dir / "index.csv").exists()
        print(f"  shard size: {os.path.getsize(out_dir / 'chunk_1.h5')/1e3:.1f} KB "
              f"for {len(U.dummy_rows())} examples\n")
        check_roundtrip(out_dir)
        check_collate_and_loss(out_dir)
        check_training_from_h5(out_dir)
    print("GATE 5 PASSED: preprocess -> HDF5 store, round-trip exact, "
          "train from store.")


if __name__ == "__main__":
    main()
