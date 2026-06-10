"""
Reusable functions for pMHC class-I dataset assembly with cluster-aware splits.

Pipeline (see processing.py for the CLI orchestration):
  Step A  cluster assignment   -> P(H) per HLA cluster
  Step B  profile construction -> length-stratified per-position AA profiles
  Step C  per-HLA-cluster sampling of synthetic pMHC pairs
  Step D  cluster-level test + 5-fold CV splits

Allele identity convention
--------------------------
The parquet column ``mhc_embedding_key`` (e.g. ``HLA-A0201``) is the canonical
allele key. It matches both the ``key`` column of ``mhc1_encodings.csv``
(-> ``mhc_sequence`` and the FASTA header used in the output ``id``) and the
member column of ``cluster_cluster.tsv`` (the HLA clustering).

All randomness flows through a single ``numpy.random.default_rng(seed)`` instance
for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_peptide_clusters(tsv_path, log=print):
    """Load the peptide-cluster membership TSV.

    Expected columns: cluster_id, representative_anchor, peptide_header,
    sequence, anchor. We only keep ``cluster_id`` and ``sequence``.

    Returns a DataFrame[cluster_id, sequence] (pyarrow-backed strings).
    """
    log(f"[load] peptide clusters: {tsv_path}")
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=["cluster_id", "sequence"],
        dtype="string[pyarrow]",
    )
    log(f"[load]   {len(df):,} membership rows, "
        f"{df['cluster_id'].nunique():,} clusters")
    return df


def build_seq_to_cluster(clusters_df, log=print):
    """dict: peptide sequence -> cluster_id (string). Deduped by sequence."""
    log("[map] building sequence -> cluster_id lookup")
    sub = clusters_df.drop_duplicates(subset="sequence")
    seqs = sub["sequence"].astype(str).to_numpy()
    cids = sub["cluster_id"].astype(str).to_numpy()
    d = dict(zip(seqs, cids))
    log(f"[map]   {len(d):,} unique sequences")
    return d


def load_hla_clusters(tsv_path, log=print):
    """Load HLA clustering (representative \\t member).

    Returns dict: representative -> list[member alleles] (includes the rep itself
    if it appears as its own member, which mmseqs output does).
    """
    log(f"[load] HLA clusters: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["rep", "member"], dtype=str)
    d = defaultdict(list)
    for rep, mem in zip(df["rep"], df["member"]):
        d[rep].append(mem)
    log(f"[load]   {len(d):,} HLA clusters, {df.shape[0]:,} memberships")
    return dict(d)


def load_mhc_encodings(csv_path, log=print):
    """Load allele key -> full MHC sequence map from mhc1_encodings.csv."""
    log(f"[load] MHC encodings: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    d = dict(zip(df["key"], df["mhc_sequence"]))
    log(f"[load]   {len(d):,} allele sequences")
    return d


# --------------------------------------------------------------------------- #
# Step A - cluster assignment
# --------------------------------------------------------------------------- #
def build_allele_pcluster_pairs(parquet_path, seq2cluster,
                                allele_col="mhc_embedding_key",
                                pep_col="long_mer", log=print):
    """Stream the parquet row-group by row-group and collect the set of distinct
    ``(allele, peptide_cluster_id)`` pairs.

    Binders and non-binders are pooled (label ignored): only the cluster id of
    each annotated peptide matters. Peptides absent from the clustering (not in
    ``seq2cluster``) and rows with a null allele are dropped.
    """
    pf = pq.ParquetFile(parquet_path)
    log(f"[stepA] streaming {pf.num_row_groups} row groups "
        f"({pf.metadata.num_rows:,} rows) from {parquet_path}")
    pairs = set()
    n_rows = 0
    n_mapped = 0
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=[allele_col, pep_col]).to_pandas()
        n_rows += len(df)
        df["cid"] = df[pep_col].map(seq2cluster)
        df = df.dropna(subset=["cid", allele_col])
        n_mapped += len(df)
        df = df.drop_duplicates([allele_col, "cid"])
        pairs.update(zip(df[allele_col].tolist(), df["cid"].tolist()))
        if (i + 1) % 5 == 0 or (i + 1) == pf.num_row_groups:
            log(f"[stepA]   row group {i + 1}/{pf.num_row_groups} | "
                f"distinct (allele,cluster) pairs so far: {len(pairs):,}")
    log(f"[stepA] done: {n_rows:,} rows, {n_mapped:,} mapped to a cluster, "
        f"{len(pairs):,} distinct (allele, peptide_cluster) pairs")
    return pairs


def pairs_to_allele2pcl(pairs):
    """dict: allele -> set(peptide_cluster_id)."""
    d = defaultdict(set)
    for a, c in pairs:
        d[a].add(c)
    return d


def _norm_key(s):
    """Normalize an allele key: drop punctuation, uppercase."""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def build_allele_resolver(parquet_alleles, encodings, mem2rep, log=print):
    """Map each parquet ``mhc_embedding_key`` to its HLA cluster representative.

    The parquet uses three incompatible spellings vs. the encoding/cluster keys,
    so we resolve in priority order:

      1. exact     - key is itself an encoding key -> its cluster.
      2. case/punct- normalized key matches exactly one encoding key
                     (e.g. ``MAMU-B05201`` -> ``Mamu-B05201``, ``H-2-KB`` ->
                     ``H-2-Kb``). One-to-one, unambiguous.
      3. precision - 2-field key (e.g. ``HLA-A0201``) is a prefix of one or more
                     full-precision encoding keys that all fall in a SINGLE HLA
                     cluster -> that cluster.

    Keys whose precision-expansion spans >1 cluster (ambiguous) or that match no
    encoding key (unresolved) are dropped. Resolution is to the *cluster* only;
    emitted alleles/sequences come from sampled cluster members downstream, so no
    representative-sequence substitution happens in the output.

    Returns (resolver: dict embedding_key -> representative, stats: dict).
    """
    enc_norm = defaultdict(list)
    for e in encodings:
        enc_norm[_norm_key(e)].append(e)
    norm_keys = list(enc_norm.keys())

    resolver = {}
    stats = defaultdict(int)
    ambiguous = []
    for k in parquet_alleles:
        if k in encodings:                                   # 1. exact
            rep = mem2rep.get(k)
            if rep is not None:
                resolver[k] = rep
                stats["exact"] += 1
            else:
                stats["no_cluster"] += 1
            continue
        nk = _norm_key(k)
        if nk in enc_norm:                                   # 2. case/punct
            reps = {mem2rep[e] for e in enc_norm[nk] if e in mem2rep}
            if len(reps) == 1:
                resolver[k] = reps.pop()
                stats["case_punct"] += 1
            elif not reps:
                stats["no_cluster"] += 1
            else:
                ambiguous.append(k)
                stats["ambiguous"] += 1
            continue
        cands = [e for ne in norm_keys if ne.startswith(nk)  # 3. precision
                 for e in enc_norm[ne]]
        reps = {mem2rep[e] for e in cands if e in mem2rep}
        if len(reps) == 1:
            resolver[k] = next(iter(reps))
            stats["precision"] += 1
        elif not reps:
            stats["unresolved"] += 1
        else:
            ambiguous.append(k)
            stats["ambiguous"] += 1

    stats = dict(stats)
    log(f"[resolve] parquet alleles: {len(parquet_alleles)} | resolved: "
        f"{len(resolver)} | {stats}")
    log(f"[resolve]   dropped {len(ambiguous)} ambiguous (multi-cluster) keys")
    return resolver, stats, ambiguous


def pairs_to_cluster_pcl(pairs, resolver, log=print):
    """Collapse distinct ``(allele, peptide_cluster)`` pairs into
    P(H): dict representative -> set(peptide_cluster_id), using ``resolver`` to
    map each parquet allele to its HLA cluster. Unresolved alleles are skipped.
    """
    ph = defaultdict(set)
    for k, c in pairs:
        rep = resolver.get(k)
        if rep is not None:
            ph[rep].add(c)
    ph = dict(ph)
    log(f"[stepA] P(H): {len(ph)} HLA clusters have >=1 annotated peptide "
        f"cluster")
    return ph


# --------------------------------------------------------------------------- #
# Step A (cont.) - borrow P(H) for isolated (empty) HLA clusters
# --------------------------------------------------------------------------- #
def _run_mafft_msa(rep_to_seq, cache_path=None, refresh_cache=False, log=print):
    """Align representative sequences with mafft (FFT-NS-2, single-threaded).

    Reproducibility: input is written in sorted-representative order and mafft is
    run with ``--thread 1`` (deterministic given the input). In addition, the
    resulting alignment is cached to ``cache_path`` keyed by a SHA-1 hash of the
    sorted (rep, sequence) pairs; a rerun with identical input reuses the cached
    MSA verbatim (and recomputes automatically if the input changes).

    rep_to_seq : dict representative -> mhc_sequence (full sequence, matching
                 how the HLA clustering was built).
    Returns (order: list[rep], aligned: dict rep -> aligned sequence upper-case).
    """
    reps = sorted(rep_to_seq)
    key = hashlib.sha1(
        "\n".join(f"{r}\t{rep_to_seq[r]}" for r in reps).encode()
    ).hexdigest()

    if cache_path and not refresh_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("key") == key:
                log(f"[borrow] reusing cached MSA ({cache_path})")
                return reps, cached["aligned"]
            log("[borrow] MSA cache key mismatch (input changed) - recomputing")
        except (OSError, ValueError):
            log("[borrow] MSA cache unreadable - recomputing")

    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        in_path = fh.name
        # mafft preserves input order; use the index as a safe header
        for i, r in enumerate(reps):
            fh.write(f">{i}\n{rep_to_seq[r]}\n")
    try:
        log(f"[borrow] running mafft MSA on {len(reps)} representative seqs")
        proc = subprocess.run(
            ["mafft", "--thread", "1", "--retree", "2", "--maxiterate", "0",
             "--quiet", in_path],
            check=True, capture_output=True, text=True,
        )
    finally:
        os.unlink(in_path)

    aligned = {}
    cur_idx = None
    chunks = []
    for line in proc.stdout.splitlines():
        if line.startswith(">"):
            if cur_idx is not None:
                aligned[reps[cur_idx]] = "".join(chunks).upper()
            cur_idx = int(line[1:].strip())
            chunks = []
        else:
            chunks.append(line.strip())
    if cur_idx is not None:
        aligned[reps[cur_idx]] = "".join(chunks).upper()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({"key": key, "aligned": aligned}, fh)
        log(f"[borrow] wrote MSA cache ({cache_path})")
    return reps, aligned


def is_excluded_nonclassical(rep):
    """True for non-classical / non-peptide-presenting genes to exclude when
    they appear as *isolated* (borrowed) clusters: MIC (MICA/MICB/MIC2), TAP
    (TAP1/TAP2), HFE, and BoLA-NC* non-classical class I.

    Care is taken not to match mouse classical alleles like ``mice-H-2Dk``
    (the substring 'MIC' inside 'MICE').
    """
    u = str(rep).upper()
    if re.search(r"MIC[AB2]", u):    # MICA / MICB / MIC2 (not 'MICE')
        return True
    if re.search(r"TAP\d", u):       # TAP1 / TAP2
        return True
    if "HFE" in u:                   # HLA-HFE
        return True
    if re.search(r"-NC\d", u):       # BoLA-NC* non-classical
        return True
    return False


def fill_empty_ph(ph, hla_clusters, encodings, exclude_fn=None,
                  msa_cache=None, refresh_cache=False, log=print):
    """Borrow P(H) for HLA clusters with no parquet annotations.

    For every HLA cluster whose P(H) is empty, align its representative sequence
    against all non-empty clusters' representatives (one shared mafft MSA) and
    copy P(H*) from the single nearest non-empty cluster (highest normalized
    identity = identical columns / columns where both reps are non-gap).

    Step C sampling is otherwise unchanged: the borrower still draws HLAs from
    its *own* member alleles, only the peptide-cluster source is borrowed.

    ``exclude_fn`` : optional predicate(rep) -> bool. Empty clusters for which it
    returns True are *excluded* entirely (not borrowed, never sampled). This is
    applied only to isolated/empty clusters; non-empty clusters of the same gene
    types (e.g. a BoLA-NC with real data) are kept and can act as donors.

    Returns (ph_filled, borrows, excluded) where borrows is
        {borrower_rep: {"donor": donor_rep, "identity": float}}
    and excluded is the sorted list of excluded isolated cluster reps.
    """
    non_empty = [r for r in ph if ph[r]]
    empty = [r for r in hla_clusters if r not in ph or not ph[r]]

    excluded = sorted(r for r in empty if exclude_fn and exclude_fn(r))
    if excluded:
        empty = [r for r in empty if r not in set(excluded)]
        log(f"[borrow] excluding {len(excluded)} isolated non-classical/MIC/TAP "
            f"clusters: {excluded}")

    if not empty:
        log("[borrow] no (remaining) empty HLA clusters - nothing to borrow")
        return dict(ph), {}, excluded
    if not non_empty:
        log("[borrow] WARNING: no non-empty clusters to borrow from")
        return dict(ph), {}, excluded

    # representatives -> full mhc sequence (skip reps without a sequence).
    # Sort the empty/donor lists so the nearest-donor argmax (and its tie-breaks)
    # is independent of dict/set iteration order -> fully reproducible borrows.
    rep_seq = {r: encodings[r] for r in (set(non_empty) | set(empty))
               if r in encodings}
    usable_empty = sorted(r for r in empty if r in rep_seq)
    usable_donors = sorted(r for r in non_empty if r in rep_seq)
    log(f"[borrow] {len(empty)} empty clusters; borrowing from "
        f"{len(usable_donors)} non-empty donors")

    _, aligned = _run_mafft_msa(
        {r: rep_seq[r] for r in set(usable_empty) | set(usable_donors)},
        cache_path=msa_cache, refresh_cache=refresh_cache, log=log)

    GAP = ord("-")
    donor_arr = np.array(
        [np.frombuffer(aligned[r].encode("ascii"), dtype=np.uint8)
         for r in usable_donors])
    donor_nongap = donor_arr != GAP

    ph_filled = dict(ph)
    borrows = {}
    for r in usable_empty:
        row = np.frombuffer(aligned[r].encode("ascii"), dtype=np.uint8)
        both = donor_nongap & (row != GAP)                 # M x L
        denom = both.sum(axis=1)
        matches = ((donor_arr == row) & both).sum(axis=1)
        ident = np.divide(matches, denom, out=np.zeros_like(matches, float),
                          where=denom > 0)
        j = int(np.argmax(ident))
        donor = usable_donors[j]
        ph_filled[r] = set(ph[donor])
        borrows[r] = {"donor": donor, "identity": round(float(ident[j]), 4)}

    ids = np.array([b["identity"] for b in borrows.values()])
    log(f"[borrow] filled {len(borrows)} clusters | identity "
        f"min={ids.min():.3f} median={np.median(ids):.3f} max={ids.max():.3f}")
    return ph_filled, borrows, excluded


# --------------------------------------------------------------------------- #
# Step B - profile construction (length-stratified column-wise frequency model)
# --------------------------------------------------------------------------- #
def _profile_from_seqs(seqs):
    """Build a length-stratified per-position amino-acid frequency profile.

    seqs : sequence of peptide strings (cluster members).

    Returns either:
      {'singleton': seq}                              (single-member cluster)
    or
      {'lengths': int array, 'length_probs': float array,
       'by_length': {L: [ (chars array, probs array), ... per position ]}}

    The 3 N-terminal and 3 C-terminal positions are captured implicitly: every
    position is modelled conditional on the peptide length, so anchor positions
    keep their length-specific distributions.
    """
    seqs = [str(s) for s in seqs]
    if len(seqs) == 1:
        return {"singleton": seqs[0]}

    lengths = np.array([len(s) for s in seqs])
    uniq_lens, counts = np.unique(lengths, return_counts=True)
    by_length = {}
    for L in uniq_lens:
        group = [s for s in seqs if len(s) == L]
        # pack equal-length strings into a (n, L) uint8 matrix for fast counting
        arr = np.frombuffer("".join(group).encode("ascii"),
                            dtype=np.uint8).reshape(len(group), int(L))
        cols = []
        for pos in range(int(L)):
            vals, cnts = np.unique(arr[:, pos], return_counts=True)
            chars = np.array([chr(v) for v in vals])
            probs = cnts / cnts.sum()
            cols.append((chars, probs))
        by_length[int(L)] = cols
    return {
        "lengths": uniq_lens.astype(int),
        "length_probs": counts / counts.sum(),
        "by_length": by_length,
    }


def build_profiles(clusters_df, needed_clusters, log=print):
    """Build profiles only for peptide clusters that appear in some P(H).

    Returns dict: cluster_id -> profile (see ``_profile_from_seqs``).
    """
    needed = set(needed_clusters)
    log(f"[stepB] building profiles for {len(needed):,} peptide clusters")
    sub = clusters_df[clusters_df["cluster_id"].isin(needed)]
    profiles = {}
    n = 0
    for cid, grp in sub.groupby("cluster_id", sort=False):
        profiles[str(cid)] = _profile_from_seqs(grp["sequence"].to_numpy())
        n += 1
        if n % 1000 == 0:
            log(f"[stepB]   {n:,} profiles built")
    n_singleton = sum(1 for p in profiles.values() if "singleton" in p)
    log(f"[stepB] done: {len(profiles):,} profiles ({n_singleton:,} singletons)")
    return profiles


# --------------------------------------------------------------------------- #
# Step C - per-HLA-cluster sampling
# --------------------------------------------------------------------------- #
def allocate_quota(n_samples, k, rng):
    """Allocate ``n_samples`` across ``k`` peptide clusters under a HARD CAP.

    The per-HLA-cluster total never exceeds ``n_samples``:
      - if ``n_samples >= k``: floor quota + uniformly-random +1 remainder, so
        every peptide cluster gets >=1 and the quotas sum to exactly n_samples.
      - if ``n_samples < k``: we cannot cover every peptide cluster within the
        cap, so we uniformly sample ``n_samples`` distinct peptide clusters and
        give each exactly 1 (the rest get 0). Total = n_samples.

    Returns a length-k int array. (Actual emitted counts may fall slightly below
    n_samples if a chosen cluster's profile can't yield enough unique peptides.)
    """
    if k == 0:
        return np.array([], dtype=int)
    if n_samples >= k:
        base = n_samples // k
        rem = n_samples % k
        quotas = np.full(k, base, dtype=int)
        if rem > 0:
            idx = rng.choice(k, size=rem, replace=False)
            quotas[idx] += 1
    else:
        quotas = np.zeros(k, dtype=int)
        idx = rng.choice(k, size=n_samples, replace=False)
        quotas[idx] = 1
    return quotas


def allocate_by_dist(q, items, probs, rng):
    """Split ``q`` integer samples across ``items`` proportional to ``probs``
    (largest-remainder with a random, fractional-part-weighted tie-break).

    Returns dict: item -> count (only positive counts).
    """
    raw = probs * q
    base = np.floor(raw).astype(int)
    rem = int(q - base.sum())
    if rem > 0:
        frac = raw - base
        if frac.sum() <= 0:
            idx = rng.choice(len(items), size=rem, replace=True)
        else:
            idx = rng.choice(len(items), size=rem, replace=True,
                             p=frac / frac.sum())
        for i in idx:
            base[i] += 1
    return {int(items[i]): int(base[i]) for i in range(len(items)) if base[i] > 0}


def sample_one_peptide(profile, length, rng):
    """Sample a single peptide of ``length`` position-by-position from its
    per-(length, position) amino-acid distributions."""
    cols = profile["by_length"][length]
    return "".join(rng.choice(chars, p=probs) for (chars, probs) in cols)


def sample_hla_cluster(rep, members, ph_clusters, profiles, encodings,
                       n_samples, rng, emitted_peptides, max_attempts=50):
    """Generate synthetic pMHC pairs for one HLA cluster (Step C).

    - Quota of peptide samples allocated across P(H) clusters under a hard cap of
      ``n_samples`` total (see ``allocate_quota``).
    - Within a cluster, the quota is split across lengths in proportion to the
      cluster's observed length distribution; peptides are sampled from the
      profile.
    - Peptides are deduped **globally**: ``emitted_peptides`` is a shared set
      mutated in place, so no peptide ever appears more than once in the entire
      dataset. (Global peptide uniqueness implies global (peptide, allele) pair
      uniqueness, so no separate pair check is needed.)
    - Alleles are sampled uniformly with replacement from members that have a
      sequence.

    Returns list of (peptide, allele, peptide_cluster_id).
    """
    valid_members = [m for m in members if m in encodings]
    pclusters = sorted(ph_clusters)
    if not valid_members or not pclusters:
        return []

    quotas = allocate_quota(n_samples, len(pclusters), rng)

    # ---- generate peptides (deduped GLOBALLY via emitted_peptides) ----
    peptides = []                       # list of (peptide, peptide_cluster_id)
    for pcl, q in zip(pclusters, quotas):
        if q <= 0:
            continue
        prof = profiles[pcl]
        if "singleton" in prof:
            seq = prof["singleton"]
            if seq not in emitted_peptides:
                emitted_peptides.add(seq)
                peptides.append((seq, pcl))
            continue
        len_alloc = allocate_by_dist(q, prof["lengths"], prof["length_probs"], rng)
        for L, qn in len_alloc.items():
            got = 0
            attempts = 0
            cap = qn * max_attempts + max_attempts
            while got < qn and attempts < cap:
                attempts += 1
                s = sample_one_peptide(prof, L, rng)
                if s in emitted_peptides:
                    continue
                emitted_peptides.add(s)
                peptides.append((s, pcl))
                got += 1

    # ---- pair each (globally-unique) peptide with a uniformly-sampled allele ----
    n_members = len(valid_members)
    return [(pep, valid_members[int(rng.integers(n_members))], pcl)
            for pep, pcl in peptides]


def build_full_dataset(ph, hla_clusters, profiles, encodings, n_samples, rng,
                       log=print):
    """Run Step C across all HLA clusters and assemble the full pair table.

    Returns (DataFrame, per_cluster_counts dict).
    Columns: peptide, mhc_seq, type, anchors, id, hla_cluster_id,
             peptide_cluster_id.
    """
    emitted_peptides = set()           # global: every peptide appears at most once
    records = []
    per_cluster_counts = {}
    reps = sorted(r for r, s in ph.items() if s)  # only HLA clusters with P(H)
    log(f"[stepC] sampling up to {n_samples} pairs/HLA-cluster (hard cap) across "
        f"{len(reps)} HLA clusters")
    for j, rep in enumerate(reps, 1):
        rows = sample_hla_cluster(rep, hla_clusters[rep], ph[rep], profiles,
                                  encodings, n_samples, rng, emitted_peptides)
        per_cluster_counts[rep] = len(rows)
        for pep, allele, pcl in rows:
            records.append((
                pep,
                encodings[allele],
                1,
                "",
                f"{allele}_{pep}_{len(pep)}",
                rep,
                pcl,
            ))
        if j % 25 == 0 or j == len(reps):
            log(f"[stepC]   {j}/{len(reps)} HLA clusters | "
                f"total pairs: {len(records):,}")
    df = pd.DataFrame.from_records(
        records,
        columns=["peptide", "mhc_seq", "type", "anchors", "id",
                 "hla_cluster_id", "peptide_cluster_id"],
    ).reset_index(drop=True)
    log(f"[stepC] done: {len(df):,} total pairs")
    return df, per_cluster_counts


# --------------------------------------------------------------------------- #
# Step D - cluster-level splits
# --------------------------------------------------------------------------- #
def make_splits(df, test_hla_frac, test_peptide_frac, cv_folds,
                cv_val_peptide_frac, rng, seed, borrows=None, excluded=None,
                hla_col="hla_cluster_id", pep_col="peptide_cluster_id",
                log=print):
    """Build the test set and 5-fold CV val sets at the cluster level.

    A pair is held out only when *both* of its axes are held out:
      test rows: hla_cluster in test-HLA  AND  peptide_cluster in test-peptide.
      fold-k val rows: hla_cluster in fold-k HLA AND peptide_cluster in
                       fold-k val-peptide.
    Pairs with only one held-out axis fall through to neither (the user's
    downstream code reconstructs training sets from cluster metadata).

    Returns (test_idx, fold_results, metadata).
    """
    df = df.reset_index(drop=True)

    all_hla = np.array(sorted(df[hla_col].unique()))
    perm = rng.permutation(len(all_hla))
    all_hla = all_hla[perm]

    # ---- test ----
    n_test_hla = round(test_hla_frac * len(all_hla))
    test_hla = set(all_hla[:n_test_hla].tolist())
    test_pep_pool = sorted(df.loc[df[hla_col].isin(test_hla), pep_col].unique())
    n_test_pep = round(test_peptide_frac * len(test_pep_pool))
    pep_perm = rng.permutation(len(test_pep_pool))
    test_pep = {test_pep_pool[i] for i in pep_perm[:n_test_pep]}
    test_mask = df[hla_col].isin(test_hla) & df[pep_col].isin(test_pep)
    test_idx = df.index[test_mask].tolist()
    log(f"[stepD] test: {len(test_hla)} HLA clusters x "
        f"{len(test_pep)} peptide clusters -> {len(test_idx):,} rows")

    # ---- 5-fold CV on the remaining HLA clusters ----
    remaining = all_hla[n_test_hla:]
    folds_hla = np.array_split(remaining, cv_folds)  # disjoint & exhaustive
    fold_results = []
    for k, fh in enumerate(folds_hla, 1):
        fh_set = set(fh.tolist())
        pool = sorted(df.loc[df[hla_col].isin(fh_set), pep_col].unique())
        n_val_pep = round(cv_val_peptide_frac * len(pool))
        pp = rng.permutation(len(pool))
        val_pep = {pool[i] for i in pp[:n_val_pep]}
        mask = df[hla_col].isin(fh_set) & df[pep_col].isin(val_pep)
        idx = df.index[mask].tolist()
        fold_results.append({
            "fold": k,
            "hla_clusters": sorted(fh_set),
            "val_peptide_clusters": sorted(val_pep),
            "val_idx": idx,
        })
        log(f"[stepD] fold {k}: {len(fh_set)} HLA clusters x "
            f"{len(val_pep)} peptide clusters -> {len(idx):,} val rows")

    metadata = {
        "seed": seed,
        "mode": "two_axis",
        "split_axis": "hla+peptide",
        "n_total_pairs": int(len(df)),
        "fractions": {
            "test_hla_frac": test_hla_frac,
            "test_peptide_frac": test_peptide_frac,
            "cv_folds": cv_folds,
            "cv_val_peptide_frac": cv_val_peptide_frac,
        },
        "test": {
            "hla_clusters": sorted(test_hla),
            "peptide_clusters": sorted(test_pep),
            "n_rows": len(test_idx),
        },
        "cv_folds_detail": [
            {
                "fold": fr["fold"],
                "hla_clusters": fr["hla_clusters"],
                "val_peptide_clusters": fr["val_peptide_clusters"],
                "n_val_rows": len(fr["val_idx"]),
            }
            for fr in fold_results
        ],
        "borrowed_assignments": borrows or {},
        "excluded_isolated_clusters": excluded or [],
    }
    return test_idx, fold_results, metadata


def make_splits_hla_only(df, test_hla_frac, cv_folds, rng, seed,
                         borrows=None, excluded=None,
                         hla_col="hla_cluster_id", log=print):
    """Single-axis (HLA-cluster only) test + k-fold CV.

    A pair is assigned purely by its HLA cluster, so there is no discarded
    "one-axis" region and every pair lands in exactly one of test / a fold:
      test         : pairs whose HLA cluster is in a random ``test_hla_frac`` of
                     all HLA clusters.
      fold-k val   : pairs of fold k's HLA clusters (remaining clusters
                     partitioned into ``cv_folds`` disjoint folds).
      fold-k train : pairs of the other folds' HLA clusters (excluding test).

    Peptide clusters are not held out. Returns (test_idx, fold_results, metadata).
    """
    df = df.reset_index(drop=True)
    all_hla = np.array(sorted(df[hla_col].unique()))
    all_hla = all_hla[rng.permutation(len(all_hla))]

    n_test = round(test_hla_frac * len(all_hla))
    test_hla = set(all_hla[:n_test].tolist())
    test_idx = df.index[df[hla_col].isin(test_hla)].tolist()
    log(f"[hla-only] test: {len(test_hla)} HLA clusters -> {len(test_idx):,} rows")

    remaining = all_hla[n_test:]
    folds_hla = np.array_split(remaining, cv_folds)
    fold_results = []
    for k, fh in enumerate(folds_hla, 1):
        fh_set = set(fh.tolist())
        idx = df.index[df[hla_col].isin(fh_set)].tolist()
        fold_results.append({"fold": k, "hla_clusters": sorted(fh_set),
                             "val_idx": idx})
        log(f"[hla-only] fold {k}: {len(fh_set)} HLA clusters -> "
            f"{len(idx):,} val rows")

    metadata = {
        "seed": seed,
        "mode": "hla_only",
        "split_axis": "hla",
        "n_total_pairs": int(len(df)),
        "fractions": {"test_hla_frac": test_hla_frac, "cv_folds": cv_folds},
        "test": {"hla_clusters": sorted(test_hla), "n_rows": len(test_idx)},
        "cv_folds_detail": [
            {"fold": fr["fold"], "hla_clusters": fr["hla_clusters"],
             "n_val_rows": len(fr["val_idx"])}
            for fr in fold_results
        ],
        "borrowed_assignments": borrows or {},
        "excluded_isolated_clusters": excluded or [],
    }
    return test_idx, fold_results, metadata


# --------------------------------------------------------------------------- #
# PMGen input preparation
# --------------------------------------------------------------------------- #
def prepare_pmgen_input(full_df):
    """Clean the assembled dataset into PMGen's expected input layout.

    - rename ``type`` -> ``mhc_type`` and ``id`` -> ``key_id`` (the original key),
    - add a new ``id`` = ``key_id`` with ``-``, ``:`` and ``*`` removed,
    - order columns: peptide, mhc_seq, mhc_type, anchors, id, hla_cluster_id,
      peptide_cluster_id, key_id.
    """
    df = full_df.rename(columns={"type": "mhc_type", "id": "key_id"})
    df["id"] = (df["key_id"].astype(str)
                .str.replace("-", "", regex=False)
                .str.replace(":", "", regex=False)
                .str.replace("*", "", regex=False))
    cols = ["peptide", "mhc_seq", "mhc_type", "anchors", "id",
            "hla_cluster_id", "peptide_cluster_id", "key_id"]
    return df[cols]


def anchor_pairs(length, min_gap=6):
    """Valid MHC-I 2-anchor combinations for a peptide of ``length``.

    1-indexed positions (i, j) with i < j and (j - i) >= ``min_gap``, generated
    nested (i ascending, then j ascending). e.g. length 8 -> ['1;7','1;8','2;8'].
    """
    return [f"{i};{j}"
            for i in range(1, length + 1)
            for j in range(i + min_gap, length + 1)]


def expand_multiple_anchors(pmgen_df, min_gap=6, log=print):
    """Expand each peptide row into one row per valid 2-anchor combination.

    The ``anchors`` column is filled with the pair (e.g. ``1;7``) and the ``id``
    is suffixed with ``_<anchor_index>`` (0-based, in generation order). ALL
    columns of ``pmgen_df`` are preserved (fixing the missing-column bug).
    """
    df = pmgen_df.copy()
    lengths = df["peptide"].str.len()
    alist = {int(L): anchor_pairs(int(L), min_gap) for L in lengths.unique()}
    df["_alist"] = lengths.map(alist)
    exp = df.explode("_alist", ignore_index=False)
    exp["anchors"] = exp["_alist"].astype(str)
    exp["id"] = exp["id"].astype(str) + "_" + \
        exp.groupby(level=0).cumcount().astype(str)
    exp = exp.drop(columns="_alist").reset_index(drop=True)
    cols = ["peptide", "mhc_seq", "mhc_type", "anchors", "id",
            "hla_cluster_id", "peptide_cluster_id", "key_id"]
    log(f"[pmgen] expanded {len(pmgen_df):,} peptides -> {len(exp):,} "
        f"anchor rows")
    return exp[cols]


def reduce_anchors(expanded_df, frac=0.5, max_anchors=4, rng=None, log=print):
    """Randomly subsample the anchor rows per original peptide.

    Grouped by ``key_id`` (unique per peptide), keep
    ``min(max_anchors, max(1, floor(frac * n_anchors)))`` rows, chosen uniformly
    at random; the original ``id`` suffixes are preserved. The cap keeps highly
    expanded peptides (e.g. 15-mers, 45 combinations) bounded. Deterministic
    given ``rng``.
    """
    rng = rng or np.random.default_rng()
    df = expanded_df.copy()
    k = df.groupby("key_id")["id"].transform("size")
    n_keep = np.maximum(1, np.floor(frac * k).astype(int))
    if max_anchors is not None:
        n_keep = np.minimum(n_keep, int(max_anchors))
    df["_r"] = rng.random(len(df))
    rank = df.groupby("key_id")["_r"].rank(method="first")
    reduced = df[rank <= n_keep].drop(columns="_r").reset_index(drop=True)
    cap = f", cap {max_anchors}" if max_anchors is not None else ""
    log(f"[pmgen] reduced anchors to <= {frac:.0%}/peptide{cap}: "
        f"{len(expanded_df):,} -> {len(reduced):,} rows")
    return reduced


# --------------------------------------------------------------------------- #
# Writing outputs
# --------------------------------------------------------------------------- #
def write_full_dataset(output_dir, full_df, log=print):
    """Write the shared full_dataset.csv (referenced by every split scheme)."""
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, "full_dataset.csv")
    full_df.to_csv(full_path, index=False)
    log(f"[write] {full_path} ({len(full_df):,} rows)")


def write_split(split_dir, test_idx, fold_results, metadata, log=print):
    """Write one split scheme: test/test.csv, cv/fold_k/val.csv, metadata.
    ``row_idx`` values index into the shared full_dataset.csv."""
    os.makedirs(split_dir, exist_ok=True)
    test_dir = os.path.join(split_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    pd.DataFrame({"row_idx": test_idx}).to_csv(
        os.path.join(test_dir, "test.csv"), index=False)
    log(f"[write] {test_dir}/test.csv ({len(test_idx):,} rows)")

    for fr in fold_results:
        fdir = os.path.join(split_dir, "cv", f"fold_{fr['fold']}")
        os.makedirs(fdir, exist_ok=True)
        pd.DataFrame({"row_idx": fr["val_idx"]}).to_csv(
            os.path.join(fdir, "val.csv"), index=False)
        log(f"[write] {fdir}/val.csv ({len(fr['val_idx']):,} rows)")

    meta_path = os.path.join(split_dir, "splits_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    log(f"[write] {meta_path}")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def log_statistics(full_df, per_cluster_counts, test_idx, fold_results,
                   borrows=None, excluded=None, log=print):
    """Log dataset statistics at the end of a run."""
    log("=" * 64)
    log("DATASET STATISTICS")
    log("=" * 64)
    log(f"  total pairs            : {len(full_df):,}")
    log(f"  unique HLA alleles     : "
        f"{full_df['id'].str.rsplit('_', n=2).str[0].nunique():,}")
    n_pep = full_df['peptide'].nunique()
    log(f"  unique peptides        : {n_pep:,}  "
        f"(duplicates: {len(full_df) - n_pep})")
    log(f"  unique HLA clusters    : {full_df['hla_cluster_id'].nunique():,}")
    log(f"  unique peptide clusters: {full_df['peptide_cluster_id'].nunique():,}")

    counts = np.array(list(per_cluster_counts.values()))
    if len(counts):
        log("  pairs per HLA cluster  : "
            f"min={counts.min()}, median={int(np.median(counts))}, "
            f"mean={counts.mean():.1f}, max={counts.max()}")

    lengths = full_df["peptide"].str.len()
    log("  peptide length distribution:")
    for L, c in lengths.value_counts().sort_index().items():
        log(f"      len {int(L):>2}: {c:,}")

    if borrows:
        ids = np.array([b["identity"] for b in borrows.values()])
        log(f"  borrowed HLA clusters  : {len(borrows):,} "
            f"(of {full_df['hla_cluster_id'].nunique():,} total)")
        log(f"  borrow identity        : min={ids.min():.3f}, "
            f"median={np.median(ids):.3f}, max={ids.max():.3f}")
        low = int((ids < 0.6).sum())
        log(f"  borrows below 0.60 id  : {low}  (low-identity = review)")
    else:
        log("  borrowed HLA clusters  : 0")
    log(f"  excluded isolated clstr: {len(excluded or [])}  "
        f"(MIC/TAP/HFE/BoLA-NC)")

    log(f"  test rows              : {len(test_idx):,}")
    for fr in fold_results:
        log(f"  fold {fr['fold']} val rows         : {len(fr['val_idx']):,}")
    log("=" * 64)
