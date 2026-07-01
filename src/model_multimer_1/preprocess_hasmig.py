"""
Parse the hasmig_mhcs dataset into the same HDF5 shard format as the main store, so
it can join training. Differences from the main preprocessor handled here:

  * input is ONE zip per id (``<id>.zip`` containing ``*_model_*.pdb`` and
    ``*_plddt.npy``), not a chunk dir;
  * the plddt .npy is PADDED — its true length is n_mhc+n_pep (from the sequences);
    the padding tail (nonsense values) is dropped;
  * there is NO PAE array — ``teacher_pae`` is stored as zeros [N,N] (this dataset is
    for the structure + pLDDT objective, not PAE);
  * per-structure scores (``pep_mean_plddt``, ``docking_score``, ``source``) from the
    CSV are carried into the H5 attrs + index so the confidence/burial FILTER can be
    applied at training time (both scores stored — no thresholds baked in here).

The PDB has the same ~200 MHC->peptide residue-number gap as the main data, so the
existing ``parse_example`` works unchanged.

Array-parallel: run --chunk c of --n-chunks N (rows strided c::N), then --merge.
See preprocess_hasmig.sbatch.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC.parent / "openfold"))     # residue_constants for parse
sys.path.insert(0, str(_SRC / "pdb"))
from parse import parse_example                        # noqa: E402


def _base_id(aid: str) -> str:
    """Strip the trailing anchor-variant index (``..._1`` -> ``...``)."""
    return re.sub(r"_\d+$", "", aid)


def process_row(r, zip_dir: Path):
    """One CSV row -> (arrays, meta) mirroring extract_example_arrays, but from a zip,
    with truncated plddt and zero PAE."""
    aid = str(r["id"])
    zp = zip_dir / f"{aid}.zip"
    if not zp.exists():
        raise FileNotFoundError(zp)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        pdb_name = next(x for x in names if x.endswith(".pdb"))
        npy_name = next(x for x in names if x.endswith("_plddt.npy"))
        pdb_bytes, npy_bytes = z.read(pdb_name), z.read(npy_name)

    tf = tempfile.NamedTemporaryFile("wb", suffix=".pdb", delete=False)
    try:
        tf.write(pdb_bytes)
        tf.close()
        ex = parse_example(tf.name, r["peptide"], r["mhc_seq"], r["anchors"],
                           r["mhc_type"], return_backbone=True)
    finally:
        os.unlink(tf.name)

    n = int(ex["aatype"].shape[0])                      # true length = n_mhc + n_pep
    plddt = np.load(io.BytesIO(npy_bytes))
    if plddt.ndim != 1 or plddt.shape[0] < n:
        raise ValueError(f"plddt shape {plddt.shape} < {n}")
    plddt = plddt[:n].astype(np.float16)                # drop the padded tail

    arrays = {
        "aatype": ex["aatype"].numpy().astype(np.uint8),
        "residue_index": ex["residue_index"].numpy().astype(np.int16),
        "anchor": ex["anchor"].numpy().astype(np.uint8),
        "segment_id": ex["segment_id"].numpy().astype(np.uint8),
        "teacher_ca": ex["teacher_ca"].numpy().astype(np.float32),
        "teacher_bb": ex["teacher_bb"].numpy().astype(np.float32),
        "teacher_plddt": plddt,
        "teacher_pae": np.zeros((n, n), np.float16),    # no PAE in this dataset
    }
    meta = {
        "n_mhc": int(ex["n_mhc"]), "n_pep": int(ex["n_pep"]),
        "mhc_type": int(r["mhc_type"]), "mhc_seq": str(r["mhc_seq"]),
        "peptide": str(r["peptide"]), "anchors": str(r["anchors"]),
        "source": str(r.get("source", "")),
        "pep_mean_plddt": float(r["pep_mean_plddt"]),
        "docking_score": float(r["docking_score"]),
    }
    return arrays, meta


def merge(out: Path):
    parts = sorted(out.glob("*.index.csv"))
    frames = []
    for p in parts:
        try:
            df = pd.read_csv(p, dtype=str)
        except pd.errors.EmptyDataError:
            continue
        if len(df):
            frames.append(df)
    if not frames:
        print(f"[merge] no non-empty index parts in {out}")
        return
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out / "index.csv", index=False)
    print(f"[merge] {len(full):,} rows across {full['shard'].nunique()} shards "
          f"-> {out/'index.csv'}")


_COLS = ["id", "base_id", "shard", "n_mhc", "n_pep", "mhc_type", "source",
         "pep_mean_plddt", "docking_score", "anchors"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--clevel", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true", help="just merge *.index.csv")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(out)
        return

    zip_dir = Path(args.zip_dir)
    df = pd.read_csv(args.csv, dtype={"id": str, "peptide": str, "mhc_seq": str,
                                      "anchors": str, "mhc_type": str, "source": str})
    rows = df.iloc[args.chunk::args.n_chunks] if args.n_chunks > 1 else df
    shard = out / f"hasmig_{args.chunk:04d}.h5"
    idxp = out / f"hasmig_{args.chunk:04d}.index.csv"
    if shard.exists() and idxp.exists() and not args.overwrite:
        print(f"[skip] {shard.name} exists (use --overwrite)")
        return

    comp = dict(compression="gzip", compression_opts=args.clevel)
    recs, ok, miss, fail = [], 0, 0, 0
    with h5py.File(shard, "w") as h5:
        for _, r in rows.iterrows():
            aid = str(r["id"])
            try:
                arrays, meta = process_row(r, zip_dir)
            except FileNotFoundError:
                miss += 1
                continue
            except Exception as e:                      # bad pdb / seq mismatch / etc.
                fail += 1
                if fail <= 25:
                    print(f"  [skip] {aid}: {str(e)[:100]}", flush=True)
                continue
            g = h5.create_group(aid)
            for k in arrays:
                g.create_dataset(k, data=arrays[k], **comp)
            base = _base_id(aid)
            for k, v in meta.items():
                g.attrs[k] = v
            g.attrs["base_id"] = base
            recs.append({"id": aid, "base_id": base, "shard": shard.name,
                         "n_mhc": meta["n_mhc"], "n_pep": meta["n_pep"],
                         "mhc_type": meta["mhc_type"], "source": meta["source"],
                         "pep_mean_plddt": meta["pep_mean_plddt"],
                         "docking_score": meta["docking_score"],
                         "anchors": meta["anchors"]})
            ok += 1
            if ok % 2000 == 0:
                print(f"  chunk {args.chunk}: {ok} ok / {miss} miss / {fail} fail",
                      flush=True)
    pd.DataFrame(recs, columns=_COLS).to_csv(idxp, index=False)
    print(f"[chunk {args.chunk}] {ok} ok, {miss} missing, {fail} failed -> {shard.name}")


if __name__ == "__main__":
    main()
