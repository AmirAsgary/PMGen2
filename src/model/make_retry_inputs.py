"""
Build balanced PMGen *retry* inputs for ids that failed preprocessing.

A failed id = an id in a chunk's ``chunk.tsv`` that is NOT in the merged HDF5
``index.csv`` (its PMGen prediction is missing/incomplete). This script:

  1. recomputes failed ids per chunk from the index (robust; no log parsing),
  2. splits them into <= --max-jobs **balanced** shards, where each shard stays
     within a single chunk (so it maps to that chunk's PMGen output dir),
  3. writes one retry TSV per shard (same columns as chunk.tsv) + a manifest the
     retry sbatch array reads, and an optional list of the incomplete per-id
     output dirs (to remove if PMGen skips ids whose folder already exists).

Run:
  python src/model/make_retry_inputs.py \
      --chunks-dir ~/projects/PMGen_2/data/pmgen_inputs/chunks \
      --h5-dir     data/processed/h5_store \
      --out-dir    data/processed/retry_inputs \
      --max-jobs 32
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd


def _chunk_num(name: str) -> int:
    return int("".join(ch for ch in name if ch.isdigit()) or 0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--h5-dir", required=True, help="store dir holding index.csv")
    ap.add_argument("--out-dir", required=True, help="where to write retry TSVs + manifest")
    ap.add_argument("--max-jobs", type=int, default=32)
    ap.add_argument("--pmgen-run-root",
                    default=f"/ptmp/{os.environ.get('USER', 'USER')}/pmgen_run",
                    help="root holding <chunk>/ PMGen output dirs (the symlink target)")
    ap.add_argument("--alphafold-subdir", default="alphafold")
    args = ap.parse_args(argv)

    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # succeeded ids per chunk, from the merged index (shard "chunk_N.h5")
    idx = pd.read_csv(Path(args.h5_dir) / "index.csv", dtype=str)
    done_by_chunk: dict[str, set] = {}
    for shard, g in idx.groupby("shard"):
        chunk = shard[:-3] if shard.endswith(".h5") else shard
        done_by_chunk[chunk] = set(g["id"])

    # failed rows per chunk = chunk.tsv ids not in the index (preserve columns)
    chunk_dirs = sorted(
        (d for d in chunks_dir.iterdir()
         if d.is_dir() and (d / "chunk.tsv").exists()),
        key=lambda d: _chunk_num(d.name))
    failed: list[tuple[str, pd.DataFrame]] = []
    total = 0
    for cdir in chunk_dirs:
        df = pd.read_csv(cdir / "chunk.tsv", sep="\t", dtype=str)
        miss = df[~df["id"].isin(done_by_chunk.get(cdir.name, set()))]
        if len(miss):
            failed.append((cdir.name, miss))
            total += len(miss)
    if total == 0:
        print("[retry] no failed ids — nothing to do.")
        return
    print(f"[retry] {total} failed ids across {len(failed)} chunks: "
          + ", ".join(f"{c}={len(m)}" for c, m in failed))

    # choose a per-shard cap so the number of shards (each within one chunk) <= max_jobs
    def n_shards(per: int) -> int:
        return sum(math.ceil(len(m) / per) for _, m in failed)

    per = max(1, math.ceil(total / args.max_jobs))
    while n_shards(per) > args.max_jobs:
        per += 1
    n = n_shards(per)
    print(f"[retry] per-shard cap {per} -> {n} balanced shards (<= {args.max_jobs})")

    # write shard TSVs + manifest + incomplete-dir list
    recs = []
    inc = open(out_dir / "incomplete_dirs.txt", "w")
    k = 0
    for chunk, miss in failed:
        output_dir = f"{args.pmgen_run_root}/{chunk}"
        af_base = f"{output_dir}/{args.alphafold_subdir}"
        for i in miss["id"]:
            inc.write(f"{af_base}/{i}\n")
        for start in range(0, len(miss), per):
            k += 1
            shard = miss.iloc[start:start + per]
            tsv = out_dir / f"retry_{k:03d}_{chunk}.tsv"
            shard.to_csv(tsv, sep="\t", index=False)
            recs.append({"shard_id": k, "chunk": chunk, "tsv": str(tsv.resolve()),
                         "n_ids": len(shard), "output_dir": output_dir})
    inc.close()
    man = pd.DataFrame(recs)
    man.to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
    print(f"[retry] wrote {k} shard TSVs + manifest.tsv "
          f"(ids/shard: min {man['n_ids'].min()}, max {man['n_ids'].max()})")
    print(f"[retry] manifest: {out_dir / 'manifest.tsv'}")
    print(f"[retry] >>> submit the retry array with:  --array=1-{k}")


if __name__ == "__main__":
    main()
