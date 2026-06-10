# How to run the pMHC dataset pipeline

Two scripts, run in order, both from the **PMGen2 project root**
(`/home/amir/amir/ParseFold/PMGen2`) using the project conda env.

```bash
PY=~/miniforge3/envs/pmbind_peptide/bin/python
cd /home/amir/amir/ParseFold/PMGen2
```

> The env must be called by absolute path — `conda activate` doesn't reliably
> switch the interpreter in this shell. The env provides pandas, pyarrow, numpy,
> matplotlib; `mafft` and `mmseqs` are used as external binaries.

---

## 1. Build the dataset + splits

```bash
$PY src/data_processing/processing.py
```

Runs in ~5 min. With all defaults it:
- maps every parquet peptide to its peptide cluster (threshold 0.5),
- reconciles parquet allele keys → HLA clusters (exact / case / 2-field precision),
- borrows P(H) for isolated HLA clusters from the nearest non-empty cluster
  (mafft MSA identity),
- samples synthetic pMHC pairs per HLA cluster,
- builds **two** cluster-level split schemes over the same dataset.

### Outputs → `data/processed/`
One shared dataset plus two split schemes (`row_idx` in every split CSV indexes
into the shared `full_dataset.csv`):

| Path | Contents |
|------|----------|
| `full_dataset.csv` | every pair: `peptide, mhc_seq, type, anchors, id, hla_cluster_id, peptide_cluster_id` (shared by both schemes) |
| `two_axis/` | **scheme A** — HLA cluster *and* peptide cluster held out (leakage-free on both axes; large discarded "one-axis" region). `test/test.csv`, `cv/fold_{1..5}/val.csv`, `splits_metadata.json`. |
| `hla_only/` | **scheme B** — HLA cluster only; no peptide-axis holdout, so every non-test pair is some fold's validation (no discards). Same file layout. |

Each `splits_metadata.json` carries `mode` (`two_axis`/`hla_only`), held-out
cluster IDs, seed, `borrowed_assignments`, and `excluded_isolated_clusters`.

### PMGen-input preparation (stage `pmgen_input_preparation`)
Runs at the end of step 1 unless `--skip-pmgen-input` is given:
| Path | Contents |
|------|----------|
| `data/processed/full_dataset_pmgeninput.csv` | cleaned dataset: `type`→`mhc_type`, original `id`→`key_id`, new `id` = key_id with `- : *` removed |
| `outputs/pmgen_input/full_dataset/Multiple_Anchors_input.tsv` | each peptide expanded into one row per valid MHC-I 2-anchor combination (1-indexed, gap ≥ `--pmgen-anchor-min-gap`=6); `anchors` filled (e.g. `1;7`), `id` suffixed `_<idx>`; **all** columns kept |
| `outputs/pmgen_input/full_dataset/Multiple_Anchors_input_reduced.tsv` | random subset keeping `min(--pmgen-max-anchors=4, floor(--pmgen-max-anchor-frac=0.5 · K))` anchor rows per peptide (≥1 each); independently seeded, so regenerable from the full TSV without a full rerun. 3,673,210 → 885,620 rows. |

### Common flags (all optional; defaults shown)
```bash
$PY src/data_processing/processing.py \
    --parquet-path data/raw/PMDb_2025_11_18_class1.parquet \
    --hla-cluster-tsv data/raw/alleles_clusters_all/cluster_cluster.tsv \
    --peptide-cluster-tsv data/raw/pepitde_clusters/anchor_all_05/clusters.tsv \
    --mhc-encodings-csv data/raw/mhc1_encodings.csv \
    --output-dir data/processed \
    --n-samples-per-hla-cluster 1000 \
    --test-hla-frac 0.10 --test-peptide-frac 0.10 \
    --cv-folds 5 --cv-val-peptide-frac 0.20 \
    --seed 42
```
`--help` lists everything. Re-running overwrites `data/processed/`.

> Notes on sampling:
> - **Hard cap:** each HLA cluster yields **at most `--n-samples-per-hla-cluster`**
>   pairs. If `|P(H)|` exceeds N, N peptide clusters are sampled (1 each); counts
>   can dip slightly below N when small/singleton clusters can't supply enough
>   unique peptides.
> - **Global peptide uniqueness:** every peptide appears at most once in the
>   entire dataset (so each row is a unique peptide-HLA pair).
> - **Exclusions:** isolated (borrowed) MIC / TAP / HFE / BoLA-NC clusters are
>   dropped (non-classical / non-peptide-presenting); recorded under
>   `excluded_isolated_clusters` in the metadata.
> - **Reproducibility:** runs are deterministic for a fixed `--seed`. The borrow
>   MSA is cached to `--msa-cache` (default `data/cache/`) and the
>   donor search is order-independent, so reruns produce byte-identical output.
>   Use `--refresh-msa-cache` to force the MSA to recompute.

---

## 2. Analysis & visualization

Run **after** step 1 (reads `full_dataset.csv`):

```bash
$PY src/data_processing/analysis.py
```

### Outputs → `analysis/processed_data_exploration/`
**Dataset-level (split-independent), at the root:**
| File | Report |
|------|--------|
| `samples_per_hla_per_cluster.csv` | per HLA cluster, sample count for each of its HLA alleles |
| `hla_cluster_contribution.csv` / `.png` | each HLA cluster's contribution (count + fraction) — bar chart |
| `peptide_cluster_sample_counts.csv` / `..._boxplot.png` | samples per peptide cluster — single box plot, all points overlaid |
| `peptide_length_distribution.csv` / `.png` | peptide length distribution of the final dataset — bar chart |

**Per split scheme, under `two_axis/` and `hla_only/`:**
| File | Report |
|------|--------|
| `split_unique_hlas.csv` / `..._pie.png` | unique HLA alleles in train/test/validation — pie |
| `split_unique_peptide_clusters.csv` / `..._pie.png` | unique peptide clusters per split — pie |
| `split_unique_pairs.csv` / `..._pie.png` | unique peptide-HLA pairs per split — pie |
| `hla_composition_by_split.csv` / `.png` | HLA-group composition (pairs + unique HLAs) per split — grouped bar (log y) |
| `peptide_length_distribution_by_subset.csv` / `.png` | length per subset (full, test, per-fold train×5 & val×5) — 12-panel figure + one CSV with a `subset` column |

The analysis asserts global peptide uniqueness (errors out on any duplicate).
Split labels: test = `test/test.csv`, validation = union of `cv/fold_*/val.csv`,
train = everything else. `train_fold_k` for the by-subset figure is reconstructed
per scheme — two_axis: clear of both test and fold-k val holdouts on both axes;
hla_only: HLA cluster not in test and not in fold-k.

> Notes: in **hla_only** every non-test pair is some fold's validation, so the
> global "train" slice is empty (full CV coverage) — the pies show test vs.
> validation. The HLA and peptide-cluster pies count membership *observed in*
> each split and can overlap across splits, so their slices may sum to more than
> the unique total; the peptide-HLA pair pie is a clean partition.

Flags: `--full-dataset`, `--output-dir`, `--dpi` (default 300).

---

## Files
- `utils.py` — all reusable functions (loading, cluster assignment, borrow,
  profiles, sampling, splits, writing, stats).
- `processing.py` — step-1 CLI.
- `analysis.py` — step-2 CLI.
