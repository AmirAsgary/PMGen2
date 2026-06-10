"""
CLI entry point: pMHC class-I dataset assembly with cluster-aware splits.

Example
-------
    ~/miniforge3/envs/pmbind_peptide/bin/python processing.py \\
        --n-samples-per-hla-cluster 1000 --seed 42

All defaults assume the script is run from the PMGen2 project root
(paths are relative to it). Outputs land under data/processed/:
a single shared ``full_dataset.csv`` plus two split schemes over it ---
``two_axis/`` (HLA cluster AND peptide cluster held out) and ``hla_only/``
(HLA cluster only) --- each with test/, cv/fold_*/ and splits_metadata.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import utils


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Assemble a pMHC class-I dataset with cluster-aware "
                    "train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet-path",
                   default="data/raw/PMDb_2025_11_18_class1.parquet")
    p.add_argument("--hla-cluster-tsv",
                   default="data/raw/alleles_clusters_all/cluster_cluster.tsv")
    p.add_argument("--peptide-cluster-tsv",
                   default="data/raw/pepitde_clusters/anchor_all_05/clusters.tsv")
    p.add_argument("--mhc-encodings-csv",
                   default="data/raw/mhc1_encodings.csv")
    p.add_argument("--output-dir",
                   default="data/processed")
    p.add_argument("--n-samples-per-hla-cluster", type=int, default=1000)
    p.add_argument("--test-hla-frac", type=float, default=0.10)
    p.add_argument("--test-peptide-frac", type=float, default=0.10)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-val-hla-frac", type=float, default=0.20,
                   help="Informational: remaining HLA clusters are partitioned "
                        "into --cv-folds disjoint exhaustive folds (~1/folds "
                        "each); this value documents that fold size.")
    p.add_argument("--cv-val-peptide-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--msa-cache",
                   default="data/cache/borrow_msa_cache.json",
                   help="cache file for the borrow-step mafft MSA "
                        "(reused across runs with identical input for "
                        "reproducibility)")
    p.add_argument("--refresh-msa-cache", action="store_true",
                   help="force recomputation of the borrow MSA cache")
    # --- pmgen_input_preparation ---
    p.add_argument("--skip-pmgen-input", action="store_true",
                   help="skip the PMGen-input preparation stage")
    p.add_argument("--pmgen-output-dir",
                   default="outputs/pmgen_input/full_dataset")
    p.add_argument("--pmgen-anchor-min-gap", type=int, default=6,
                   help="minimum index gap between the two MHC-I anchors")
    p.add_argument("--pmgen-max-anchor-frac", type=float, default=0.5,
                   help="max fraction of anchor rows kept per peptide in the "
                        "reduced file")
    p.add_argument("--pmgen-max-anchors", type=int, default=4,
                   help="absolute cap on anchor rows kept per peptide in the "
                        "reduced file (applied with --pmgen-max-anchor-frac)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    log("pMHC dataset assembly starting")
    log(f"args: {vars(args)}")
    rng = np.random.default_rng(args.seed)

    # ---- load inputs ----
    clusters_df = utils.load_peptide_clusters(args.peptide_cluster_tsv, log=log)
    seq2cluster = utils.build_seq_to_cluster(clusters_df, log=log)
    hla_clusters = utils.load_hla_clusters(args.hla_cluster_tsv, log=log)
    encodings = utils.load_mhc_encodings(args.mhc_encodings_csv, log=log)

    # ---- Step A: cluster assignment ----
    pairs = utils.build_allele_pcluster_pairs(
        args.parquet_path, seq2cluster, log=log)
    # Resolve parquet embedding keys -> HLA cluster representative, reconciling
    # the three allele spellings (exact / case-punct / 2-field precision).
    mem2rep = {m: rep for rep, members in hla_clusters.items() for m in members}
    parquet_alleles = {a for a, _ in pairs}
    resolver, _resolver_stats, _ambiguous = utils.build_allele_resolver(
        parquet_alleles, encodings, mem2rep, log=log)
    ph = utils.pairs_to_cluster_pcl(pairs, resolver, log=log)
    # Borrow P(H) for isolated (empty) HLA clusters from the nearest non-empty
    # cluster by representative-sequence identity (mafft MSA). HLAs are still
    # sampled from the borrower's own members; only peptide clusters are borrowed.
    # Isolated MIC/TAP/HFE/BoLA-NC (non-classical, non-peptide-presenting)
    # clusters are excluded entirely rather than borrowed.
    ph, borrows, excluded = utils.fill_empty_ph(
        ph, hla_clusters, encodings,
        exclude_fn=utils.is_excluded_nonclassical,
        msa_cache=args.msa_cache, refresh_cache=args.refresh_msa_cache, log=log)

    # ---- Step B: profiles (only for clusters reachable in some P(H)) ----
    needed = set().union(*ph.values()) if ph else set()
    profiles = utils.build_profiles(clusters_df, needed, log=log)
    del seq2cluster  # free the big lookup before sampling

    # ---- Step C: per-HLA-cluster sampling ----
    full_df, per_cluster_counts = utils.build_full_dataset(
        ph, hla_clusters, profiles, encodings,
        args.n_samples_per_hla_cluster, rng, log=log)

    # ---- write the shared dataset once ----
    utils.write_full_dataset(args.output_dir, full_df, log=log)

    # ---- Step D: TWO split schemes over the same dataset ----
    # (A) two-axis (HLA cluster AND peptide cluster held out) -> two_axis/
    log("[splitD] === scheme A: two-axis (HLA+peptide) ===")
    ta_test, ta_folds, ta_meta = utils.make_splits(
        full_df,
        test_hla_frac=args.test_hla_frac,
        test_peptide_frac=args.test_peptide_frac,
        cv_folds=args.cv_folds,
        cv_val_peptide_frac=args.cv_val_peptide_frac,
        rng=rng, seed=args.seed, borrows=borrows, excluded=excluded, log=log,
    )
    utils.write_split(os.path.join(args.output_dir, "two_axis"),
                      ta_test, ta_folds, ta_meta, log=log)

    # (B) HLA-only (single-axis) -> hla_only/
    log("[splitD] === scheme B: HLA-only ===")
    ho_test, ho_folds, ho_meta = utils.make_splits_hla_only(
        full_df,
        test_hla_frac=args.test_hla_frac,
        cv_folds=args.cv_folds,
        rng=rng, seed=args.seed, borrows=borrows, excluded=excluded, log=log,
    )
    utils.write_split(os.path.join(args.output_dir, "hla_only"),
                      ho_test, ho_folds, ho_meta, log=log)

    # ---- pmgen_input_preparation ----
    if not args.skip_pmgen_input:
        log("[pmgen] === PMGen input preparation ===")
        pmgen_df = utils.prepare_pmgen_input(full_df)
        pmgen_csv = os.path.join(args.output_dir, "full_dataset_pmgeninput.csv")
        pmgen_df.to_csv(pmgen_csv, index=False)
        log(f"[pmgen] wrote {pmgen_csv} ({len(pmgen_df):,} rows)")

        os.makedirs(args.pmgen_output_dir, exist_ok=True)
        expanded = utils.expand_multiple_anchors(
            pmgen_df, min_gap=args.pmgen_anchor_min_gap, log=log)
        full_tsv = os.path.join(args.pmgen_output_dir,
                                "Multiple_Anchors_input.tsv")
        expanded.to_csv(full_tsv, sep="\t", index=False)
        log(f"[pmgen] wrote {full_tsv} ({len(expanded):,} rows)")

        # dedicated, independently-seeded RNG so the reduction is reproducible
        # on its own (regenerable from the full TSV without a full pipeline run)
        reduced = utils.reduce_anchors(
            expanded, frac=args.pmgen_max_anchor_frac,
            max_anchors=args.pmgen_max_anchors,
            rng=np.random.default_rng(args.seed), log=log)
        red_tsv = os.path.join(args.pmgen_output_dir,
                               "Multiple_Anchors_input_reduced.tsv")
        reduced.to_csv(red_tsv, sep="\t", index=False)
        log(f"[pmgen] wrote {red_tsv} ({len(reduced):,} rows)")

    # ---- stats (dataset-level + per-scheme test sizes) ----
    utils.log_statistics(full_df, per_cluster_counts, ta_test, ta_folds,
                         borrows=borrows, excluded=excluded, log=log)
    log(f"[stats] two-axis  test={len(ta_test):,} "
        f"val(union)={sum(len(f['val_idx']) for f in ta_folds):,}")
    log(f"[stats] hla-only  test={len(ho_test):,} "
        f"val(union)={sum(len(f['val_idx']) for f in ho_folds):,}")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
