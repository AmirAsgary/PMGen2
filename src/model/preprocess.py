"""
One-time preprocessing: PMGen teacher chunks -> streamable HDF5 store.

Reads each chunk dir (``<chunks-dir>/chunk_*/chunk.tsv`` + the per-id PMGen
outputs under ``chunk_*/<output-link>/<alphafold-subdir>/<id>/``) and writes one
HDF5 shard per chunk (group per anchor id) plus a merged ``index.parquet``. Each
chunk is independent, so jobs can run in parallel on disjoint ``--chunks`` and the
run is resumable (existing shards are skipped unless ``--overwrite``).

Train/val/test are then just id lists over this single store (no duplication):
``train.py --h5-dir <out> --scheme two_axis --fold 1`` (or ``--dummy``).

Example
-------
  ~/miniforge3/envs/pmgen2/bin/python src/model/preprocess.py \\
      --chunks-dir ~/projects/PMGen_2/data/pmgen_inputs/chunks \\
      --out-dir data/processed/h5_store
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks-dir", default=None,
                   help="dir containing chunk_*/chunk.tsv (+ output symlink); "
                        "required unless --merge-only")
    p.add_argument("--out-dir", required=True, help="output HDF5 store dir")
    p.add_argument("--merge-only", action="store_true",
                   help="just merge existing per-chunk *.index.csv in --out-dir "
                        "into index.csv (run after a parallel array finishes)")
    p.add_argument("--chunks", default=None,
                   help="comma-separated subset of chunk dir names (for parallel "
                        "jobs), e.g. chunk_1,chunk_2")
    p.add_argument("--alphafold-subdir", default="alphafold")
    p.add_argument("--output-link", default="output",
                   help="symlink/dir under each chunk pointing at its PMGen run")
    p.add_argument("--compression", default="gzip",
                   choices=["gzip", "lzf", "none"])
    p.add_argument("--clevel", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-merge", action="store_true",
                   help="skip merging per-chunk indices (do it later)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if args.merge_only:
        U.merge_indices(Path(args.out_dir), log=log)
        return
    if args.chunks_dir is None:
        raise SystemExit("--chunks-dir is required unless --merge-only")

    chunks = args.chunks.split(",") if args.chunks else None
    U.preprocess_chunks(
        chunks_dir=Path(args.chunks_dir), out_dir=Path(args.out_dir),
        chunks=chunks, overwrite=args.overwrite, merge=not args.no_merge,
        alphafold_subdir=args.alphafold_subdir, output_link=args.output_link,
        compression=(None if args.compression == "none" else args.compression),
        clevel=args.clevel, log=log,
    )


if __name__ == "__main__":
    main()
