# Methods: structural domain boundary inference from large-scale Foldseek all-vs-all structural alignments

## 1. Overview

Domain boundaries for proteins in a non-redundant, clustered subset of the
AlphaFold Database (AFDB; *n* = 2.3 × 10⁶ structures) were inferred from
an all-vs-all Foldseek structural alignment search performed across the
full structure set. The underlying rationale is that structurally
independent domains recur as discrete, self-contained units across
evolution: when a query protein shares a domain with many other proteins
in the database, the aligned region of those structural matches will
repeatedly cover the same sub-interval of the query, regardless of what
lies on either side of it in the full-length protein. Conversely, linker
or inter-domain regions are covered far less consistently, since they are
not under the same structural constraint and do not co-occur with the same
fixed partner regions. This produces a characteristic, locally variable
profile of alignment coverage along the length of a query protein, with
plateaus corresponding to candidate domains and valleys corresponding to
candidate linker/boundary regions. Domain boundaries were called
computationally by detecting this plateau–valley structure in the
per-residue coverage profile, independently for each of the 2.3 × 10⁶
query proteins in the database.

## 2. Input data

All-vs-all structural alignments were generated previously using Foldseek
on the full set of 2.3 × 10⁶ non-redundant AFDB structures. The alignment
output (3.66 GB, 30,045,247 alignment records) was provided in tab-separated
format with the following fields: `query`, `target`, `fident`, `alnlen`,
`mismatch`, `gapopen`, `qstart`, `qend`, `tstart`, `tend`, `evalue`, `bits`,
`lddt`, `alntmscore`. Records were grouped contiguously by `query` in the
source file (verified computationally prior to analysis; see Section 3.1),
consistent with standard Foldseek search output where all hits for a given
query are emitted together.

## 3. Computational pipeline

### 3.1 Pre-processing and validation

Prior to domain calling, the assumption that all alignment records sharing
a given query identifier are contiguous within the file was verified by a
single sequential linear scan over all 30,045,247 records, tracking
whether any query identifier reappeared after the file had moved on to a
different query. This check passed with zero violations, confirming that
the file could be safely partitioned into independent byte-range segments
without splitting any query's alignment records across segment boundaries
— a precondition for correct parallel processing in Section 3.3.

The 3.66 GB alignment file was then partitioned into 64 contiguous
byte-range segments of approximately equal size, with each partition
boundary constrained to fall between two records belonging to different
queries (never within a single query's record block). This partitioning
was performed by a single linear pass over the file, deciding cut points
using only the `query` field, without parsing or loading numeric fields,
and required no prior sorting of the input.

### 3.2 Quality filtering of alignment records

Within each query's set of alignment records, individual hits were
retained as evidence for domain inference only if they satisfied all of
the following criteria:

- structural alignment TM-score (`alntmscore`) ≥ 0.4
- alignment e-value ≤ 1 × 10⁻²

Self-alignments (`query` = `target`) were excluded from the evidence set
used for domain calling but were used to determine query sequence length
(taken as the maximum `qend` value observed across all records for that
query, including the self-alignment).

### 3.3 Coverage profile construction

For each query protein independently, a per-residue, TM-score–weighted
coverage depth profile was constructed over the full length of the query.
For every retained alignment hit *i*, with aligned query interval
[`qstart`ᵢ, `qend`ᵢ] and weight wᵢ = `alntmscore`ᵢ, the weight wᵢ was added
to every residue position within [`qstart`ᵢ, `qend`ᵢ]. The resulting raw
depth profile was smoothed with a moving-average filter (window size = 5
residues) to reduce sensitivity to single-residue noise at alignment
boundaries arising from minor variation in individual structural
superpositions.

### 3.4 Domain boundary calling

Within the smoothed coverage profile, a position was classified as
belonging to a candidate domain block if its smoothed depth exceeded 20%
of the maximum depth observed anywhere in that query's profile (the
"valley threshold"); contiguous runs of such positions define candidate
domain blocks, separated by valleys (positions falling below threshold,
interpreted as candidate linker or inter-domain regions). Candidate blocks
separated by a gap of 5 residues or fewer were merged, since gaps of this
size are attributable to alignment-boundary noise rather than genuine
linker regions.

For each candidate block exceeding a minimum length of 20 residues, the
set of alignment hits substantially overlapping that block was identified
(an alignment hit was assigned to a block if at least 60% of the hit's own
aligned length fell within the block's coordinates). A domain was called
only if at least 3 independent alignment hits were assigned to its
candidate block (the minimum support threshold), and the domain's final
reported boundaries were taken as the median `qstart` and median `qend`
across all assigned hits — rather than the candidate block's own
coverage-derived extent — in order to obtain boundary estimates that are
robust to noise in any single structural alignment. Domains for which the
resulting median-based length fell below the 20-residue minimum after this
robust re-estimation were discarded.

### 3.5 Parallel execution

Domain calling (Sections 3.2–3.4) was performed independently and in
parallel across the 64 file partitions described in Section 3.1, as a
SLURM job array (one array task per partition, 1 CPU core and 4 GB memory
per task). Because alignment records for any given query are guaranteed
to lie entirely within a single partition, no query's evidence set was
split across parallel tasks, and per-query results required no
cross-partition reconciliation. Each task processed its partition by
streaming alignment records sequentially, buffering only the alignment
records belonging to the query currently being read, and emitting domain
calls for that query as soon as the next record's query identifier
changed — bounding peak memory per task by the largest single query's
hit count rather than by partition size. Outputs from all 64 tasks were
subsequently concatenated, with a defensive check confirming that no query
identifier appeared in the output of more than one partition (which would
indicate a violation of the partitioning precondition).

## 4. Parameter summary

All thresholds below were applied uniformly across all 2.3 × 10⁶ query
proteins.

| Parameter | Value | Role |
|---|---|---|
| Minimum `alntmscore` | 0.4 | Minimum structural alignment quality for a hit to be used as domain evidence |
| Maximum e-value | 1 × 10⁻² | Minimum statistical significance for a hit to be used as domain evidence |
| Coverage smoothing window | 5 residues | Moving-average window applied to the coverage profile |
| Valley threshold | 20% of profile maximum | Depth fraction below which a position is treated as a candidate boundary/linker |
| Block merge gap | 5 residues | Maximum gap between candidate blocks before they are merged into one |
| Minimum domain length | 20 residues | Minimum length for a candidate region to be considered a domain |
| Minimum overlap fraction | 60% | Minimum fraction of a hit's aligned length that must fall within a block for the hit to support that block |
| Minimum support | 3 independent hits | Minimum number of supporting hits required to call a domain |

## 5. Output

For each query protein with at least one called domain, the pipeline
reports: the query identifier; the called domain's start and end
coordinates (1-based, inclusive, in query sequence coordinates); the
domain length; the number of independent alignment hits supporting the
call; the mean `alntmscore` of those supporting hits; the full query
length; and the total number of alignment hits observed for that query
prior to quality filtering. Proteins receiving more than one called
domain are reported with one row per domain.

## 6. Validation

Prior to execution on the full dataset, the algorithm described in
Sections 3.2–3.4 was validated against synthetic alignment data
constructed with known ground-truth domain architectures, comprising:
a single-domain protein; a two-domain protein with a defined linker
region; a three-domain protein; and a single-domain protein subjected to
a substantial fraction of low-quality, randomly distributed spurious
alignment hits (intended to test robustness of the quality-filtering step
in Section 3.2). In all four cases, the called domain boundaries matched
the known ground truth, with boundary coordinates differing from the true
values by at most a few residues — consistent with the expected precision
limit given that domain/linker transitions are not themselves sharply
defined at single-residue resolution, even in experimentally characterized
structures. Robustness to spurious hits, to partial/truncated alignments,
and to a worst-case query with several thousand supporting hits (single
largest tested cluster: 11,001 alignment records for one query) was
additionally confirmed.

## 7. Software and reproducibility

The pipeline was implemented in Python 3 (NumPy for numerical operations;
no other third-party dependencies). All scripts, default parameters, and
the synthetic validation dataset are available in the project repository
(`domain_pipeline/`); see `README.md` for full deployment and parameter
documentation, and `domain_caller.py` for the reference implementation of
the algorithm described in Sections 3.2–3.4.

## 8. Caveats and limitations

The valley-detection approach is sensitive only to domain boundaries that
manifest as a reduction in cross-protein structural alignment coverage at
the boundary. Domains that are *always* observed fused to an adjacent
domain across all available homologs in the database (i.e., no homolog in
the searched set possesses one domain without the other) cannot be
separated by this method and will be reported as a single, longer domain;
this is an inherent limitation of inferring domain boundaries from
homology coverage patterns rather than from independent structural or
biophysical criteria (e.g., contact-map-based domain parsing on the
individual structure). Called domain boundaries should therefore be
interpreted as evidence-supported structural-homology units rather than
as a definitive, exhaustive domain decomposition, and the `n_support`
and `mean_weight` fields in the output should be used to assess
confidence on a per-call basis.

