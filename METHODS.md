# Methods

## Dataset assembly and cluster-aware splitting

We assembled a synthetic peptide–MHC class I (pMHC-I) dataset with cluster-aware
train/validation/test partitions from the PMDb (2025-11-18) class I collection,
comprising 43,864,880 peptide–allele annotations (binders and non-binders pooled,
as only cluster membership, not binding label, was used). Each peptide
(`long_mer`) was assigned to a peptide cluster via an exact match against an
anchor-based clustering computed at a normalised similarity threshold of 0.5
(9,991 clusters over 11,231,758 peptides), and each annotation's allele was
mapped to an HLA cluster defined by MMseqs2 `easy-cluster` (`--min-seq-id 0.85
-c 0.85 --cov-mode 1`) over the full MHC sequences of 30,577 alleles (282
clusters). Because the source table encodes alleles in a compressed two-field
key (`mhc_embedding_key`) whereas the clustering uses full IMGT-style keys, keys
were reconciled in priority order — exact, case/punctuation-normalised, and
two-field-to-full-precision prefix matching that resolved to a single cluster —
yielding 444 of 479 distinct alleles (multi-cluster-ambiguous keys were dropped).
For each HLA cluster H we then defined its accessible peptide-cluster repertoire
P(H) as the union of the peptide clusters of all peptides annotated to any of H's
member alleles. HLA clusters with no annotations (isolated clusters) inherited
the repertoire of their nearest annotated cluster, determined from a single
MAFFT multiple-sequence alignment (FFT-NS-2) of all 282 representative sequences
and the resulting normalised pairwise identity (identical columns divided by
columns at which both representatives are non-gap); isolated clusters
corresponding to non-classical or non-peptide-presenting genes (MIC, TAP, HFE,
and BoLA-NC; 13 clusters) were excluded rather than imputed. For every retained
peptide cluster we built a length-stratified, position-specific amino-acid
frequency profile (per peptide length, an independent categorical distribution at
each position), reducing to the stored sequence for singleton clusters. We then
generated synthetic pMHC-I pairs per HLA cluster (N = 1,000, hard-capped): the
quota was distributed evenly across the clusters of P(H) (uniform random
remainder), within each cluster apportioned across peptide lengths in proportion
to that cluster's observed length distribution, and peptides were drawn
position-by-position from the corresponding profile; every generated peptide was
required to be globally unique (a peptide appears at most once in the entire
dataset, which in turn guarantees globally unique peptide–HLA pairs), and each
peptide was paired with an allele drawn uniformly at random from its HLA
cluster's member alleles. We then produced two complementary cluster-level split
schemes over the same dataset. In the *two-axis* scheme, holdouts are
leakage-free along both axes: the test set comprises pairs whose HLA cluster lies
in a random 10 % of HLA clusters *and* whose peptide cluster lies in a random
10 % of the peptide clusters appearing in those HLA clusters, and five-fold
cross-validation on the remaining HLA clusters (partitioned into five disjoint
folds) holds out 20 % of each fold's peptide clusters; pairs with only one
held-out axis are assigned to neither set (reconstructed as fold-specific
training data from the recorded cluster membership). In the *HLA-only* scheme,
partitioning is by HLA cluster alone (test = a random 10 % of clusters, the
remainder split into five disjoint folds), so every non-test pair serves as
validation in exactly one fold with no discarded region. All stochastic steps
used a single seeded NumPy generator (`default_rng`, seed 42); the borrow
alignment is cached and the nearest-donor search is order-independent, making
runs byte-for-byte reproducible. The final dataset contained 267,551 pairs
(267,551 unique peptides, zero duplicates) spanning 269 HLA clusters (226 of
which borrowed their repertoire) and 9,396 peptide clusters; the two-axis scheme
yielded 2,241 test and 47,017 validation pairs, whereas the HLA-only scheme
yielded 26,978 test and 240,573 cross-validation pairs.

Finally, for input to the downstream structure-based pipeline (PMGen), the
dataset was reformatted — renaming the binding-type and key columns and deriving
a punctuation-free identifier — and each peptide was expanded into its admissible
MHC-I anchor configurations, defined as every ordered pair of 1-indexed anchor
positions separated by at least six residues. This yields (L−6)(L−5)/2
configurations for a peptide of length L (e.g. three for an 8-mer: 1;7, 1;8,
2;8), giving 3,673,210 peptide–anchor rows in total. To curb this expansion, a
reproducible post-processing step randomly retained
min(4, ⌊0.5·K⌋) configurations per peptide (at least one) — i.e. at most half of
the K configurations, capped at four to bound highly expanded long peptides —
reducing the set to 885,620 rows.
