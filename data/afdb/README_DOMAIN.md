# Domain-calling pipeline for Foldseek all-vs-all AFDB alignments
Genarated here: `/cbscratch/amirasgary2/Bind/domain_extractor/domain_pipeline`.
Finds candidate domain boundaries within query proteins by looking for
recurring aligned sub-regions (`qstart`-`qend`) across many structural
homology hits. If many independent hits all align to roughly the same
sub-range of a query, that sub-range is probably a structurally
independent domain.

## How it works (short version)

For each query: build a per-residue "coverage depth" profile from all its
good-quality hits (weighted by `alntmscore`). Domains show up as plateaus in
this profile, separated by valleys (linker regions rarely covered by any
single hit). Each plateau's boundaries are reported as the **median**
`qstart`/`qend` of the hits supporting it (robust to individual noisy
alignments), along with a support count (how many independent hits back
that call) so you can filter low-confidence calls later.

See `domain_caller.py` docstring for the full algorithmic detail.

## Files

| File | Purpose | Run as |
|---|---|---|
| `verify_grouping.py` | Checks that all rows for a query are contiguous in the file. **Run this first.** | single srun/sbatch |
| `chunk_finder.py` | Finds N safe byte-offset boundaries (never splits a query's rows). | single srun/sbatch |
| `worker.py` | Calls domains for all queries in one byte-range chunk. | one per SLURM array task |
| `domain_caller.py` | Core algorithm (imported by `worker.py`, not run directly). | n/a (library) |
| `merge_results.py` | Concatenates all partial outputs, sanity-checks, prints summary stats. | single srun/sbatch |
| `run_stage0_prep.sh` | sbatch script wrapping verify + chunk-finding. | `sbatch` |
| `run_stage1_array.sh` | sbatch **array** script wrapping the worker. | `sbatch` |
| `run_stage2_merge.sh` | sbatch script wrapping the merge. | `sbatch` |
| `test_data/` | Synthetic data with known ground truth, used to validate the algorithm before trusting it on real data. |  |

## Why this design (chunking strategy)

Your file is already grouped by query (all of `AF-A0A2N5KJK7...`'s hits
together, then it moves to the next query). That means we can split the file
into N independent byte ranges **for free** — no sorting needed — as long as
we're careful to cut only at a boundary between two different queries, never
in the middle of one query's hit list. `chunk_finder.py` does exactly that in
one O(n) sequential pass over the file (string operations only, no float
parsing, so it's fast even at 3.3GB).

`verify_grouping.py` checks the contiguity assumption actually holds on your
real file before you commit to this strategy — if Foldseek's output for some
reason interleaves a query's hits with another's (shouldn't happen with
standard `--format-output` and default sorting, but worth checking once),
the script tells you and exits with an error rather than silently producing
wrong chunk boundaries.

Within a chunk, `worker.py` streams line-by-line and buffers only the
*current* query's hits in memory, flushing (calling domains + writing
output) as soon as the query id changes. So peak memory is bounded by the
single largest query cluster in your database, not by chunk size. I
stress-tested this with a synthetic query carrying 11,000 hits and it ran
instantly with negligible memory.

## Step-by-step deployment

1. **Copy these files to your cluster** (e.g. into your home or project dir).
   Edit the `PIPELINE_DIR` and `INPUT` variables at the top of each of the
   three `run_stage*.sh` scripts to match your actual paths.

2. **Stage 0 — verify + chunk** (fast, single job):
   ```bash
   mkdir -p logs
   sbatch run_stage0_prep.sh
   ```
   Wait for it to finish, then check the log:
   ```bash
   cat logs/prep_<jobid>.out
   ```
   It should say `OK: grouping assumption HOLDS`. If instead it reports
   violations, **stop** — do not proceed to Stage 1 with this chunking
   strategy. Either pre-sort the file by query (`sort -k1,1`, expensive but
   one-time) or come back and we adjust the approach.

   Then inspect `chunks.json`:
   ```bash
   python3 -c "import json; print(json.load(open('chunks.json'))['n_chunks_actual'])"
   ```
   Update `#SBATCH --array=0-<that number minus 1>` in
   `run_stage1_array.sh` to match.

3. **Stage 1 — parallel domain calling** (the main array job):
   ```bash
   mkdir -p partials
   sbatch run_stage1_array.sh
   ```
   Check progress with `squeue -u $USER`. Each task's stderr log
   (`logs/worker_<jobid>_<taskid>.err`) prints a one-line summary when done:
   lines processed, queries processed, domains called, elapsed time.

4. **Stage 2 — merge**:
   ```bash
   sbatch run_stage2_merge.sh
   ```
   or, to auto-chain it after Stage 1 finishes (recommended so you don't
   have to babysit it):
   ```bash
   ARRAY_JOBID=$(sbatch --parsable run_stage1_array.sh)
   sbatch --dependency=afterok:${ARRAY_JOBID} run_stage2_merge.sh
   ```

5. **Result**: `final_domains.tsv`, columns:

   | column | meaning |
   |---|---|
   | `query` | query protein id (AFDB accession) |
   | `domain_start`, `domain_end` | called domain boundaries (1-based, inclusive, in query coordinates) |
   | `domain_len` | `domain_end - domain_start + 1` |
   | `n_support` | number of independent hits supporting this domain call — **use this to filter for confidence** |
   | `mean_weight` | mean `alntmscore` of supporting hits |
   | `qlen` | full query length |
   | `n_total_hits_seen` | total hits seen for this query (before quality filtering) — useful to distinguish "no domain called because no good hits exist" from "no domain called because hits didn't cluster" |

## Validating before you trust it on your real data

Run the included sanity check against synthetic data with known ground
truth (single-domain, two-domain, three-domain, and noisy-single-domain
proteins):
```bash
python3 test_data/test_against_ground_truth.py
```
Expected output recovers all ground-truth boundaries (within a few residues
— exact boundaries are inherently fuzzy even biologically, since the precise
domain/linker transition residue isn't a sharply defined thing). I'd
recommend running this on your actual cluster too, just to confirm the
environment (numpy version etc.) behaves identically.

## Tuning parameters (passed to `worker.py`, set in `run_stage1_array.sh`)

| Parameter | Default | What it controls |
|---|---|---|
| `--min-alntmscore` | 0.4 | Hits below this structural similarity are not used as domain evidence. Lower = more sensitive but noisier; the algorithm is fairly robust to noise (tested with 200 garbage hits mixed into 60 good ones and still got the right answer), but very low thresholds (e.g. <0.2) will start admitting near-random matches. |
| `--max-evalue` | 1e-2 | Additional significance filter. |
| `--min-support` | 3 | Minimum independent hits required to call a domain. Raise this if you want only well-supported, common domains (e.g. for building a "high-confidence domain dictionary"); lower it if you specifically want to catch rare/lineage-specific domains that may only have a handful of homologs in the 2.3M-entry clustered set. |
| `--min-domain-len` | 20 | Minimum residues. Foldseek/TM-align alignments below ~20-30 residues are often unreliable, so don't go much lower without also tightening `--min-alntmscore`. |
| `--valley-frac` | 0.2 | A position counts as a "valley" (domain boundary) if smoothed coverage drops below this fraction of the query's peak coverage. Raise it (e.g. 0.3-0.4) to be more aggressive about splitting into more, smaller domains; lower it (e.g. 0.1) to be more conservative and merge weakly-separated regions into one domain. |
| `--smooth-window` | 5 | Moving-average window (residues) applied to the coverage profile before valley-finding. Larger = smoother profile, less sensitive to single-hit boundary noise, but coarser resolution on closely-spaced domains. |
| `--min-overlap-frac` | 0.6 | When computing the robust median boundary for a candidate domain, a hit counts as "supporting" it only if this fraction of the hit's own aligned length falls inside the candidate block. |

If you get implausible results (e.g. almost every protein gets called as
"one domain = whole protein," or conversely everything gets shattered into
many tiny pieces), the first two knobs to revisit are `--valley-frac` and
`--min-support` — I'd suggest running Stage 1 on a small held-out subset
first (e.g. just chunk 0) with a couple of parameter settings and eyeballing
results on proteins where you already know the expected domain
architecture (e.g. via Pfam/InterPro), before committing to the full
2.3M-query run.

## Resource sizing notes (honest caveats)

I benchmarked the core domain-calling algorithm itself at roughly 9,000
queries/sec single-core (pure compute, no I/O) — so for 2.3M queries the
*algorithm* would take well under 10 minutes total even on one core. This
means the job is **I/O/parsing-bound, not CPU-bound**: most of the wall
time will be spent reading and tab-splitting 3.3GB of text, not in the
domain-calling logic itself.

I could not benchmark realistic I/O throughput against your actual cluster
filesystem (I only have a small sandbox to test correctness, not your
storage's real performance), so treat the `--time=04:00:00` in
`run_stage1_array.sh` as a deliberately generous safety margin rather than a
tight estimate — with 64-way parallelism each task only has to read ~50MB,
which should comfortably finish in minutes even on modest I/O, but cluster
filesystems vary a lot (especially under concurrent load from 64
simultaneous array tasks all reading the same file). If tasks are
finishing in, say, under 5 minutes consistently, feel free to shrink
`--time` and `--mem` for faster queue turnaround on reruns; if any task is
struggling, check whether concurrent reads from 64 tasks against one file
are contending on your filesystem (you could reduce `--n-chunks` in Stage 0,
or stagger array task starts with `--array=0-63%16` to cap how many run
concurrently).
