"""
Visualization & analysis of the assembled pMHC dataset (full_dataset.csv).

Produces three reports under
    analysis/processed_data_exploration/

  1. samples_per_hla_per_cluster.csv
       For each HLA cluster, how many samples each of its HLA alleles has.
       Columns: hla_cluster_id, allele, n_samples.

  2. hla_cluster_contribution.{csv,png}
       How much each HLA cluster contributes to the final dataset
       (bar chart, sorted descending). CSV columns:
       hla_cluster_id, n_samples, fraction.

  3. peptide_cluster_sample_counts.csv + ..._boxplot.png
       How many samples come from each peptide cluster — a single box plot
       (white box, black borders/median) with every point overlaid.
       CSV columns: peptide_cluster_id, n_samples.

Run with the project conda env, e.g.
    ~/miniforge3/envs/pmbind_peptide/bin/python analysis.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_COLORS = {"train": "#4C72B0", "test": "#DD8452", "validation": "#55A868"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyse / visualise the assembled pMHC dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--full-dataset",
                   default="data/processed/full_dataset.csv")
    p.add_argument("--output-dir",
                   default="analysis/processed_data_exploration")
    p.add_argument("--processed-dir",
                   default="data/processed",
                   help="dir holding test/test.csv and cv/fold_*/val.csv")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def _extract_allele(id_series):
    """id = '<allele>_<peptide>_<length>'; allele has no underscores."""
    return id_series.str.rsplit("_", n=2).str[0]


def group_allele(key):
    """Map an allele key to a broad group. Mirrors
    src/clean_hlas/utils.py:group_allele so this composition report is directly
    comparable to analysis/raw_data_exploration/hla_composition.*."""
    key = str(key)
    if key.startswith("HLA-A"):
        return "HLA-A"
    elif key.startswith("HLA-B"):
        return "HLA-B"
    elif key.startswith("HLA-C"):
        return "HLA-C"
    elif key.startswith("HLA-E"):
        return "HLA-E"
    elif key.startswith("HLA-F"):
        return "HLA-F"
    elif key.startswith("HLA-G"):
        return "HLA-G"
    elif key.startswith("Mamu"):
        return "Mamu"
    elif key.startswith("SLA"):
        return "SLA"
    elif key.startswith("Gogo"):
        return "Gogo"
    elif key.startswith("Patr"):
        return "Patr"
    elif key.startswith(("BOLA", "BoLA", "BolA")):
        return "BOLA"
    elif key.startswith("Eqca"):
        return "Eqca"
    elif key.startswith("H-2") or key.startswith("mice"):
        return "H2"
    else:
        return "Other"


def report_samples_per_hla(df, out_dir):
    """(1) For each HLA cluster, sample count per HLA allele."""
    tbl = (
        df.groupby(["hla_cluster_id", "allele"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["hla_cluster_id", "n_samples"],
                     ascending=[True, False])
    )
    path = os.path.join(out_dir, "samples_per_hla_per_cluster.csv")
    tbl.to_csv(path, index=False)
    print(f"[1] wrote {path}  ({len(tbl):,} cluster-allele rows)")
    return tbl


def report_hla_cluster_contribution(df, out_dir, dpi):
    """(2) Contribution of each HLA cluster to the dataset (csv + bar chart)."""
    counts = df["hla_cluster_id"].value_counts()
    contrib = pd.DataFrame({
        "hla_cluster_id": counts.index,
        "n_samples": counts.values,
        "fraction": counts.values / counts.values.sum(),
    })
    csv_path = os.path.join(out_dir, "hla_cluster_contribution.csv")
    contrib.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(contrib)), contrib["n_samples"].values,
           color="#4C72B0", edgecolor="none")
    ax.set_xlabel(f"HLA cluster (n={len(contrib)}, sorted by contribution)")
    ax.set_ylabel("Number of samples (pMHC pairs)")
    ax.set_title("HLA cluster contribution to the final dataset")
    ax.set_xlim(-1, len(contrib))
    ax.margins(x=0)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "hla_cluster_contribution.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] wrote {csv_path} and {png_path}")
    return contrib


def report_peptide_cluster_boxplot(df, out_dir, dpi, rng_seed=42):
    """(3) Per-peptide-cluster sample counts: csv + single box plot w/ points."""
    counts = (
        df["peptide_cluster_id"].value_counts()
        .rename_axis("peptide_cluster_id")
        .reset_index(name="n_samples")
        .sort_values("n_samples", ascending=False)
    )
    csv_path = os.path.join(out_dir, "peptide_cluster_sample_counts.csv")
    counts.to_csv(csv_path, index=False)

    vals = counts["n_samples"].values
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.boxplot(
        vals,
        positions=[1],
        widths=0.5,
        patch_artist=True,
        showfliers=False,                       # all points drawn manually below
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    rng = np.random.default_rng(rng_seed)
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(np.ones(len(vals)) + jitter, vals,
               s=8, alpha=0.35, color="#333333", zorder=3)
    ax.set_xticks([1])
    ax.set_xticklabels(["peptide clusters"])
    ax.set_ylabel("Number of samples in final dataset")
    ax.set_title(f"Samples per peptide cluster (n={len(vals):,})")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "peptide_cluster_sample_counts_boxplot.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] wrote {csv_path} and {png_path}")
    return counts


def label_splits(df, processed_dir):
    """Tag each row 'train' / 'test' / 'validation'.

    test       = rows in test/test.csv
    validation = rows in any cv/fold_*/val.csv (folds are disjoint)
    train      = everything else (reconstructed downstream from cluster metadata)
    Returns the df with a new 'split' column.
    """
    n = len(df)
    test_idx = set(pd.read_csv(
        os.path.join(processed_dir, "test", "test.csv"))["row_idx"])
    val_idx = set()
    for f in sorted(glob.glob(os.path.join(processed_dir, "cv", "fold_*",
                                           "val.csv"))):
        val_idx |= set(pd.read_csv(f)["row_idx"])
    split = np.array(["train"] * n, dtype=object)
    split[list(test_idx)] = "test"
    split[list(val_idx)] = "validation"
    df = df.copy()
    df["split"] = split
    print(f"[split] train={int((split=='train').sum()):,} "
          f"test={len(test_idx):,} validation={len(val_idx):,}")
    return df


def verify_no_duplicate_peptides(df):
    """Assert global peptide uniqueness; report the result."""
    n, u = len(df), df["peptide"].nunique()
    if n == u:
        print(f"[check] peptide uniqueness OK - {u:,} peptides, 0 duplicates")
    else:
        raise AssertionError(
            f"[check] DUPLICATE PEPTIDES: {n - u:,} duplicate rows "
            f"({u:,} unique of {n:,})")


def _pie(counts_by_split, title, csv_path, png_path, dpi):
    """Render a 3-slice train/test/validation pie + save its data CSV."""
    order = ["train", "test", "validation"]
    labels = [s for s in order if s in counts_by_split]
    values = [int(counts_by_split[s]) for s in labels]
    pd.DataFrame({"split": labels, "count": values}).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=[f"{s}\n{v:,}" for s, v in zip(labels, values)],
        colors=[SPLIT_COLORS[s] for s in labels],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white"),
    )
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[pie] wrote {csv_path} and {png_path}")


def report_split_pies(df, out_dir, dpi):
    """(4-6) Pie charts of unique HLAs / peptide clusters / peptide-HLA pairs
    across train / test / validation."""
    g = df.groupby("split")
    # (4) unique HLA alleles per split
    hlas = g["allele"].nunique().to_dict()
    _pie(hlas, "Unique HLA alleles per split",
         os.path.join(out_dir, "split_unique_hlas.csv"),
         os.path.join(out_dir, "split_unique_hlas_pie.png"), dpi)
    # (5) unique peptide clusters per split
    pcl = g["peptide_cluster_id"].nunique().to_dict()
    _pie(pcl, "Unique peptide clusters per split",
         os.path.join(out_dir, "split_unique_peptide_clusters.csv"),
         os.path.join(out_dir, "split_unique_peptide_clusters_pie.png"), dpi)
    # (6) unique peptide-HLA pairs per split (rows are unique pairs)
    pairs = g.apply(lambda s: s[["peptide", "allele"]].drop_duplicates().shape[0],
                    include_groups=False).to_dict()
    _pie(pairs, "Unique peptide-HLA pairs per split",
         os.path.join(out_dir, "split_unique_pairs.csv"),
         os.path.join(out_dir, "split_unique_pairs_pie.png"), dpi)


def report_hla_composition_by_split(df, out_dir, dpi):
    """(7) Final HLA-group composition per split (post-exclusion/processing),
    comparable to analysis/raw_data_exploration. Reports both pair counts and
    unique-allele counts; saves a wide CSV + a grouped bar chart."""
    df = df.copy()
    df["allele_group"] = df["allele"].map(group_allele)
    splits = ["train", "test", "validation"]

    pairs = pd.crosstab(df["allele_group"], df["split"])
    uniq = (df.groupby(["allele_group", "split"])["allele"].nunique()
            .unstack(fill_value=0))

    out = pd.DataFrame(index=pairs.index)
    for s in splits:
        out[f"{s}_pairs"] = pairs.get(s, 0)
    out["total_pairs"] = out[[f"{s}_pairs" for s in splits]].sum(axis=1)
    for s in splits:
        out[f"{s}_unique_hlas"] = uniq.get(s, 0)
    out = (out.sort_values("total_pairs", ascending=False)
           .reset_index().rename(columns={"index": "allele_group"}))
    csv_path = os.path.join(out_dir, "hla_composition_by_split.csv")
    out.to_csv(csv_path, index=False)

    # grouped bar chart of pair counts per split (log y: train >> test)
    groups = out["allele_group"].tolist()
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups)), 6))
    for i, s in enumerate(splits):
        ax.bar(x + (i - 1) * w, out[f"{s}_pairs"].values, w,
               label=s, color=SPLIT_COLORS[s])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("Number of pairs (log scale)")
    ax.set_title("HLA-group composition per split (post-exclusion)")
    ax.legend(title="split")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "hla_composition_by_split.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[7] wrote {csv_path} and {png_path}")
    return out


def report_peptide_length_distribution(df, out_dir, dpi):
    """(8) Peptide length distribution of the final dataset (csv + bar chart)."""
    lengths = df["peptide"].str.len()
    counts = lengths.value_counts().sort_index()
    out = pd.DataFrame({
        "length": counts.index.astype(int),
        "count": counts.values,
        "percentage": counts.values / counts.values.sum() * 100,
    })
    csv_path = os.path.join(out_dir, "peptide_length_distribution.csv")
    out.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(out["length"].astype(str), out["count"].values,
           color="#4C72B0", edgecolor="black")
    for x, c in zip(range(len(out)), out["count"].values):
        ax.text(x, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Peptide length (residues)")
    ax.set_ylabel("Number of peptides")
    ax.set_title(f"Peptide length distribution (n={len(df):,})")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "peptide_length_distribution.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[8] wrote {csv_path} and {png_path}")
    return out


def report_length_distribution_by_subset(df, processed_dir, out_dir, dpi):
    """(9) Peptide length distribution per subset in one framed figure:
    full dataset, test, and per-fold train (x5) and validation (x5).

    Subset definitions (train reconstruction depends on the split scheme):
      full          - all rows
      test          - rows in test/test.csv
      val_fold_k    - rows in cv/fold_k/val.csv
      train_fold_k  - two_axis: HLA cluster AND peptide cluster both clear of the
                      test holdout AND of fold-k's val holdout (leakage-free).
                      hla_only: HLA cluster not in test and not in fold-k.
    One combined CSV (a 'subset' column distinguishes the rows) + one PNG.
    """
    with open(os.path.join(processed_dir, "splits_metadata.json")) as fh:
        meta = json.load(fh)
    mode = meta.get("mode", "two_axis")
    length = df["peptide"].str.len()
    hla = df["hla_cluster_id"]
    pep = df["peptide_cluster_id"]
    n = len(df)

    test_hla = set(meta["test"]["hla_clusters"])
    if mode == "hla_only":
        clear_of_test = ~hla.isin(test_hla)
    else:
        test_pep = set(meta["test"]["peptide_clusters"])
        clear_of_test = ~hla.isin(test_hla) & ~pep.isin(test_pep)

    test_idx = pd.read_csv(
        os.path.join(processed_dir, "test", "test.csv"))["row_idx"]
    test_mask = pd.Series(False, index=df.index)
    test_mask.iloc[test_idx] = True

    # ordered: (label, boolean mask)
    subsets = [("full", pd.Series(True, index=df.index)),
               ("test", test_mask)]
    fold_masks = []
    for fold in sorted(meta["cv_folds_detail"], key=lambda f: f["fold"]):
        k = fold["fold"]
        fh_hla = set(fold["hla_clusters"])
        if mode == "hla_only":
            train_mask = clear_of_test & ~hla.isin(fh_hla)
        else:
            fv_pep = set(fold["val_peptide_clusters"])
            train_mask = clear_of_test & ~hla.isin(fh_hla) & ~pep.isin(fv_pep)
        vidx = pd.read_csv(os.path.join(processed_dir, "cv", f"fold_{k}",
                                        "val.csv"))["row_idx"]
        val_mask = pd.Series(False, index=df.index)
        val_mask.iloc[vidx] = True
        fold_masks.append((k, train_mask, val_mask))

    all_lengths = sorted(length.unique())

    # ---- combined CSV ----
    rows = []
    def add(label, mask):
        c = length[mask].value_counts().reindex(all_lengths, fill_value=0)
        tot = int(c.sum())
        for L in all_lengths:
            rows.append({"subset": label, "length": int(L),
                         "count": int(c[L]),
                         "percentage": (100 * c[L] / tot) if tot else 0.0})
    add("full", subsets[0][1])
    add("test", subsets[1][1])
    for k, tr, va in fold_masks:
        add(f"train_fold_{k}", tr)
        add(f"val_fold_{k}", va)
    csv_path = os.path.join(out_dir, "peptide_length_distribution_by_subset.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- framed figure: rows = [full/test, then 5 folds], cols = [left,right] ----
    panels = [("full", subsets[0][1], "#4C72B0"),
              ("test", subsets[1][1], SPLIT_COLORS["test"])]
    for k, tr, va in fold_masks:
        panels.append((f"train_fold_{k}", tr, SPLIT_COLORS["train"]))
        panels.append((f"val_fold_{k}", va, SPLIT_COLORS["validation"]))

    fig, axes = plt.subplots(6, 2, figsize=(11, 18), sharex=True)
    axes = axes.ravel()
    xpos = np.arange(len(all_lengths))
    for ax, (label, mask, color) in zip(axes, panels):
        c = length[mask].value_counts().reindex(all_lengths, fill_value=0)
        ax.bar(xpos, c.values, color=color, edgecolor="black")
        ax.set_title(f"{label}  (n={int(c.sum()):,})", fontsize=10)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(int(L)) for L in all_lengths])
    for ax in axes[-2:]:
        ax.set_xlabel("Peptide length (residues)")
    for r in range(6):
        axes[r * 2].set_ylabel("Number of peptides")
    fig.suptitle(f"Peptide length distribution by subset (total n={n:,})",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png_path = os.path.join(out_dir, "peptide_length_distribution_by_subset.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[9] wrote {csv_path} and {png_path}")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[load] {args.full_dataset}")
    df = pd.read_csv(args.full_dataset,
                     usecols=["peptide", "id", "hla_cluster_id",
                              "peptide_cluster_id"])
    df["allele"] = _extract_allele(df["id"])
    print(f"[load]   {len(df):,} rows | {df['hla_cluster_id'].nunique()} HLA "
          f"clusters | {df['peptide_cluster_id'].nunique()} peptide clusters")
    verify_no_duplicate_peptides(df)

    # ---- dataset-level reports (split-independent): output_dir root ----
    report_samples_per_hla(df, args.output_dir)
    report_hla_cluster_contribution(df, args.output_dir, args.dpi)
    report_peptide_cluster_boxplot(df, args.output_dir, args.dpi)
    report_peptide_length_distribution(df, args.output_dir, args.dpi)

    # ---- per split-scheme reports: output_dir/<scheme>/ ----
    for scheme in ("two_axis", "hla_only"):
        split_dir = os.path.join(args.processed_dir, scheme)
        if not os.path.isdir(split_dir):
            print(f"[skip] no split scheme at {split_dir}")
            continue
        out_sub = os.path.join(args.output_dir, scheme)
        os.makedirs(out_sub, exist_ok=True)
        print(f"\n=== scheme: {scheme} ===")
        dfs = label_splits(df, split_dir)
        report_split_pies(dfs, out_sub, args.dpi)
        report_hla_composition_by_split(dfs, out_sub, args.dpi)
        report_length_distribution_by_subset(df, split_dir, out_sub, args.dpi)
    print("done.")


if __name__ == "__main__":
    main()
