# PDB → HDF5 pipeline (sequence + per-residue pLDDT + sparse CA-CA distances)
This data is created under cbscratch under `/cbscratch/amirasgary2/Bind/plddt_extractor` path.
Extracts amino acid sequence, per-residue pLDDT (B-factor column), and the
sparse sub-8Å CA-CA distance matrix from ~2.5M AlphaFold-style PDB files, in
parallel on your SLURM cluster, into a single HDF5 file that streams
cleanly into PyTorch for training a sequence → pLDDT (and/or structure)
model.

## Files

| File | Purpose |
|---|---|
| `pdb_parse.py` | Core parser: PDB file → (sequence, per-residue pLDDT, CA coordinates) + sparse CA-CA distance computation. Imported by `process_chunk.py`. |
| `make_chunks.py` | Splits `pdb_ids.txt` into K chunk files for the array job. |
| `process_chunk.py` | Worker: parses one chunk's PDB files, writes one HDF5 file for that chunk. |
| `run_chunks.sbatch` | SLURM array job: runs `process_chunk.py` once per chunk, in parallel. |
| `check_chunks_complete.py` | Verifies every chunk produced a valid output before combining. |
| `combine_chunks.py` | Merges all per-chunk HDF5 files into one final HDF5 file. |
| `run_combine.sbatch` | SLURM batch job wrapping `combine_chunks.py`. |
| `torch_dataset.py` | PyTorch `Dataset` + `collate_fn` + smoke-test script for the final HDF5 file. |

## Dependencies

On the cluster: `python3`, `numpy`, `h5py`, `scipy` (and `torch` for the
training side, not needed for the extraction steps below). No BioPython
needed — the parser is plain fixed-width PDB column parsing, which is much
faster than a full structure parser at this file count. `scipy` is used
specifically for `scipy.spatial.cKDTree`, to compute the sparse CA-CA
distance matrix without ever materializing a dense LxL matrix per protein.

Check what you already have:
```bash
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import h5py; print(h5py.__version__)"
python3 -c "import scipy; print(scipy.__version__)"
```

If any of these are missing, you have two options:

**A) Install into your current env**, if you have write access to it:
```bash
pip install numpy h5py scipy --break-system-packages
```

**B) Create a dedicated conda env** (recommended if your current env is a
shared/read-only `base`, which is common on clusters):
```bash
conda create -n plddt python=3.11 numpy h5py scipy -y
```
Then tell the sbatch scripts to use it — no file editing needed:
```bash
sbatch --array=$(cat outputs/chunks/array_range.txt) \
       --export=CONDA_ENV_NAME=plddt \
       run_chunks.sbatch

sbatch --export=CONDA_ENV_NAME=plddt run_combine.sbatch
```
Both `run_chunks.sbatch` and `run_combine.sbatch` will `conda activate
plddt` automatically when `CONDA_ENV_NAME` is set this way, and print which
`python3` they ended up using at the top of the log so you can confirm it
picked up the right one. Leave `CONDA_ENV_NAME` unset (the default) to just
use whatever `python3` already resolves to on the compute node.

## Your actual paths

```
DATA_DIR   = /cbscratch/amirasgary2/Bind/data/large/alphafold_db    # has pdb/ and pdb_ids.txt
CODE_DIR   = /cbscratch/amirasgary2/Bind/plddt_extractor             # these scripts live here
OUTPUT_DIR = /cbscratch/amirasgary2/Bind/plddt_extractor/outputs     # everything this pipeline writes
```

`run_chunks.sbatch` and `run_combine.sbatch` already default to these paths
(via `${DATA_DIR:-...}` etc.), so you normally don't need to pass `--export`
at all on this cluster — only do so if you want to point at a different
location for a one-off run.

## Step 1 — chunk the id list

Run this on the login/head node (it's cheap, no need to submit it as a job):

```bash
cd /cbscratch/amirasgary2/Bind/plddt_extractor
mkdir -p outputs/logs

python3 make_chunks.py \
    --pdb-ids /cbscratch/amirasgary2/Bind/data/large/alphafold_db/pdb_ids.txt \
    --out-dir outputs/chunks \
    --k 200
```

Pick K based on cluster capacity, not just file count — K is your degree of
parallelism, not a tuning knob for chunk size. A few hundred is a reasonable
start for 2.5M files; too many tiny array tasks adds SLURM scheduling
overhead, too few makes each task run a long time and any single failure
costs more to re-run. This prints the **actual** number of chunks created
(can be slightly less than K if K doesn't divide evenly) and writes:
- `outputs/chunks/manifest.tsv` — chunk_idx → chunk filename → id count
- `outputs/chunks/array_range.txt` — the exact `--array=` range to use, e.g. `0-199`
- `outputs/chunks/chunk_0000.txt`, `outputs/chunks/chunk_0001.txt`, ... — the actual id lists

## Step 2 — pilot one chunk, then submit the array job

**Pilot first.** I can't tell you real per-file throughput on your
filesystem from outside it — reading ~2.5M small files is usually
I/O-bound (cold-cache reads dominate over actual parsing), and the new
sparse CA-CA distance computation (a KD-tree build + neighbor query per
protein) adds some extra CPU work per file on top of that, scaling with
protein length. Run array index 0 by itself and check the rate it logs
every 5000 files:

```bash
cd /cbscratch/amirasgary2/Bind/plddt_extractor
sbatch --array=0 run_chunks.sbatch
# once it finishes:
tail outputs/logs/pdb2h5_*_0.out
```

That log line (`N/M done (X files/s, ...)`) tells you the real rate. Use it
to sanity check `--time` for a full chunk (`chunk_size / rate`), then submit
the whole array, overriding `--array` from `array_range.txt` so you never
have to hand-edit the file:

```bash
sbatch --array=$(cat outputs/chunks/array_range.txt) run_chunks.sbatch
```

To use a distance cutoff other than the 8Å default, override `DISTANCE_CUTOFF`:
```bash
sbatch --array=$(cat outputs/chunks/array_range.txt) \
       --export=DISTANCE_CUTOFF=10.0 \
       run_chunks.sbatch
```

Each array task:
- looks up its chunk file via `outputs/chunks/manifest.tsv`,
- runs a single-threaded `for` loop over that chunk's PDB files inside
  `process_chunk.py`,
- writes `outputs/chunk_h5/chunk_XXXX.h5`,
- skips re-processing if that file already exists (safe to resubmit, e.g.
  after fixing failures — see Step 3).

Parallelism comes from running many array tasks concurrently, not from
multithreading inside a task — this matches what you asked for (for-loop
per job, jobs run in parallel via sbatch).

Adjust `--cpus-per-task` / `--mem` / `--time` / `--partition` at the top of
`run_chunks.sbatch` (or override on the `sbatch` command line) once you've
seen the pilot's real throughput. `--mem=4G --cpus-per-task=2` per task is a
safe starting guess for a few-thousand to ~10k files/chunk window.

## Step 3 — verify completeness before combining

```bash
cd /cbscratch/amirasgary2/Bind/plddt_extractor
python3 check_chunks_complete.py --chunk-dir outputs/chunks --out-dir outputs/chunk_h5
```

With hundreds of array tasks over 2.5M files, some will likely fail
(OOM-kill, node issue, time limit). This script tells you exactly which
chunk indices to resubmit:

```bash
sbatch --array=3,17,42 run_chunks.sbatch
```

Per-file parse failures (a single bad PDB inside an otherwise-fine chunk)
don't fail the whole chunk — they're logged to
`outputs/chunk_h5/chunk_XXXX.h5.failed.txt` and skipped, so
`process_chunk.py` keeps going.

## Step 4 — combine into the final training file

```bash
cd /cbscratch/amirasgary2/Bind/plddt_extractor
sbatch run_combine.sbatch
```

Or chain it directly after the array job finishes, without manually waiting:

```bash
ARRAY_JOB_ID=$(sbatch --parsable --array=$(cat outputs/chunks/array_range.txt) run_chunks.sbatch)
sbatch --dependency=afterok:${ARRAY_JOB_ID} run_combine.sbatch
```
(Note: `afterok` only fires combine if every array task exits 0. If any task
hits its `--time` limit or gets OOM-killed, combine won't auto-run — that's
intentional, since you should run Step 3's check first anyway.)

This produces `outputs/plddt_dataset.h5`. It's a two-pass merge (read all
chunk sizes first, pre-allocate, then stream-copy each chunk into its
slice), so its memory footprint stays roughly proportional to **one**
chunk's size, not the full dataset — safe even with very large K x chunk-size.

## HDF5 layout (why it streams well in PyTorch)

Per-protein data is **not** stored as one HDF5 group/dataset per protein
(that would mean millions of tiny datasets — terrible HDF5 metadata
overhead and slow random access). Instead, everything is flattened:

```
/pdb_id      (N,)              fixed-width bytes, e.g. b"AF-A0A009E921-F1-model_v4"
/seq_offset  (N,)   int64      start index of protein i in /sequence and /plddt
/seq_length  (N,)   int32      length of protein i
/sequence    (R,)   uint8      all sequences concatenated, amino acids as token ids
/plddt       (R,)   float16    all per-residue pLDDT concatenated

/dist_offset (N,)   int64      start index of protein i in /dist_i, /dist_j, /dist_value
/dist_count  (N,)   int32      number of CA-CA pairs kept for protein i
/dist_i      (P,)   int32      row index of each kept pair, LOCAL to that protein (0..length-1)
/dist_j      (P,)   int32      column index of each kept pair (i < j always), LOCAL
/dist_value  (P,)   float16    CA-CA distance in Angstrom for that pair
```

where N = number of proteins, R = total residues across all proteins, and
P = total number of sub-cutoff CA-CA pairs across all proteins.

To read protein `i`'s sequence/pLDDT: `start, length = seq_offset[i], seq_length[i]`,
then slice `sequence[start:start+length]` and `plddt[start:start+length]`.
This is O(1) random access per protein with no ragged/object dtypes, which
is what makes `torch_dataset.py`'s `__getitem__` cheap and memory-safe
regardless of dataset size.

**The sparse CA-CA distance matrix**: only residue pairs `(i, j)` with
`i < j` and CA-CA distance **strictly less than 8 Ångström** are stored —
everything else (the vast majority of any LxL distance matrix) is implicitly
zero and never written anywhere. This is what "sparse" means here: instead
of an `L x L` dense matrix per protein, you get a short list of `(i, j, distance)`
triples — typically only ~5% of all possible pairs survive an 8Å cutoff on a
typical protein chain, so this is also a large storage win, not just a
formatting choice.

To read protein `i`'s distance pairs:
```python
dstart, dcount = dist_offset[i], dist_count[i]
pair_i = dist_i[dstart:dstart+dcount]      # LOCAL indices, 0..seq_length[i]-1
pair_j = dist_j[dstart:dstart+dcount]
pair_d = dist_value[dstart:dstart+dcount]  # distances in Angstrom
```
`dist_i`/`dist_j` are indices **local to protein `i`'s own sequence**
(0-based, 0..length-1), not global indices into `/sequence` — a CA-CA
contact only ever connects two residues of the same protein, so this keeps
the indices small (no need for int64) and means you can use them directly
once you've sliced out that protein's own sequence/pLDDT, with no extra
arithmetic. The distance matrix is symmetric and the diagonal is always
zero, so only the `i < j` half is stored; reconstruct the full symmetric
matrix yourself if you need it (see `to_dense_distance_matrix` in
`torch_dataset.py`) — only do this for individual proteins you actually
need densified, not for the whole dataset.

If a residue has no resolved CA coordinate at all (rare; only happens with
a malformed source PDB missing a CA atom line for that residue), it's
simply excluded from every pair — there's no special sentinel value to
watch for, that residue's index just never appears in `dist_i`/`dist_j`.

Amino acid encoding (`AA_ALPHABET` in `process_chunk.py` / `torch_dataset.py`):
`"ACDEFGHIKLMNPQRSTVWYXBZJU"`, 1-indexed (0 reserved as a pad token for
collation). Includes the 20 standard amino acids plus X/B/Z/J for
unknown/ambiguous codes seen in real PDB files (e.g. `UNK`, `ASX`, `MSE` →
mapped to the nearest standard residue, see `THREE_TO_ONE` in `pdb_parse.py`).

**Note on precision**: pLDDT and CA-CA distance are both stored as
`float16` (not `float32`/`float64`) to keep disk size down at this scale.
For pLDDT (range 0–100) max rounding error is ~0.03; for distance (range
0–8Å) it's smaller still, both far below any model's prediction error. If
you'd rather not take that tradeoff for either, change the relevant
`np.float16` to `np.float32` in both `process_chunk.py` and
`combine_chunks.py` (and in `torch_dataset.py`'s `__getitem__` if you want
full precision through to the tensor) — doubles that dataset's disk
footprint; sequence storage and the int32 index arrays are unaffected.

The cutoff itself (`distance_cutoff_angstrom`, default 8.0) is recorded as
an attribute on both the per-chunk and final HDF5 files, and is also
configurable per run via `process_chunk.py --distance-cutoff` /
`run_chunks.sbatch`'s `DISTANCE_CUTOFF` variable, in case you want to
regenerate with a different threshold later without touching the code.

## Using it in PyTorch

```python
from torch_dataset import PlddtH5Dataset, collate_fn, to_dense_distance_matrix
from torch.utils.data import DataLoader

ds = PlddtH5Dataset("/cbscratch/amirasgary2/Bind/plddt_extractor/outputs/plddt_dataset.h5")
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8,
                 collate_fn=collate_fn, persistent_workers=True)

for batch in dl:
    batch["sequence"]    # LongTensor (B, T_max), 0 = pad
    batch["plddt"]       # FloatTensor (B, T_max)
    batch["mask"]        # BoolTensor (B, T_max), True = real residue
    batch["length"]      # LongTensor (B,)
    batch["edge_index"]  # LongTensor (2, total_edges), indices into flattened (B*T_max,) nodes
    batch["edge_dist"]   # FloatTensor (total_edges,), CA-CA distance per edge
    batch["edge_batch"]  # LongTensor (total_edges,), which batch item each edge belongs to

# For one un-batched sample, you can densify its sparse distances into a
# regular LxL matrix if your model wants that (e.g. for a small protein):
sample = ds[0]
dense = to_dense_distance_matrix(sample["edge_index"], sample["edge_dist"], sample["length"])
# dense.shape == (length, length), symmetric, 0.0 wherever there was no sub-8A pair
```

Or run the smoke test directly:

```bash
python3 torch_dataset.py \
    --h5 /cbscratch/amirasgary2/Bind/plddt_extractor/outputs/plddt_dataset.h5 \
    --num-workers 4 --batch-size 16
```

The dataset never opens its HDF5 file handle until first accessed **inside
the process that will use it** — this is deliberate and required for
correctness with `num_workers > 0`: an HDF5 file handle opened before
`fork()` and then used concurrently from multiple forked processes is not
safe. Each DataLoader worker lazily opens its own private handle on first
`__getitem__` call (see `_ensure_open` / `__getstate__` in `torch_dataset.py`).

## What I verified locally before handing this to you

I don't have h5py/torch/network access in my sandbox, so most of this was
validated using a stub HDF5 backend and hand-built synthetic PDB files
(including a deliberately malformed one). `scipy` WAS available locally,
so the distance-matrix math itself was checked against a real brute-force
reference, not just a stub:

- **Parsing**: confirmed correct sequence + pLDDT extraction including a
  residue whose atom lines are out of the usual N/CA/C/O order, and caught
  & fixed a real bug where fixed-column slicing alone would silently
  mis-read the B-factor on a file with narrower-than-expected coordinate
  fields. The parser now cross-checks the column-sliced B-factor against a
  whitespace-tokenized read of the same region and skips (rather than
  silently corrupts) any line where they disagree.
- **Chunking**: verified no IDs are lost or duplicated across chunks for
  both evenly- and unevenly-divisible K, and that K > N is handled.
- **Per-chunk → combined HDF5**: verified the full round trip — multiple
  chunks with different protein counts and different pdb_id string
  widths — reconstructs every protein's exact sequence and pLDDT (modulo
  expected float16 rounding), with correct local→global offset shifting
  during the merge.
- **CA coordinate extraction**: same column-misalignment defense as pLDDT
  (whitespace cross-check) applied independently to the x/y/z fields;
  verified correct extraction even when a residue's atoms are listed out
  of the usual N/CA/C/O order, and that a residue missing a CA line
  entirely is recorded as NaN rather than corrupting or dropping it (which
  would otherwise misalign sequence/pLDDT/distance indices).
- **Sparse CA-CA distance computation**: verified the KD-tree-based
  extraction (`scipy.spatial.cKDTree`) against a brute-force `cdist`
  reference on a realistic random-walk protein chain — exact match on
  which pairs survive the cutoff and their distances. Verified the strict
  "< 8Å" semantics (scipy's own cutoff is inclusive, "≤"; this needed an
  explicit extra filter) at the exact boundary (8.0Å excluded, 7.9999Å
  included). Verified NaN-coordinate residues are excluded from every pair
  without crashing or corrupting neighboring indices, via explicit
  index-remapping after filtering. Verified the chunk→combine merge keeps
  `dist_offset` correctly shifted by a running PAIR count (independent
  from the residue-count cursor used for `seq_offset`), while `dist_i`/
  `dist_j` are correctly left UNshifted (they're local to each protein).
- **The actual sbatch script bodies** (bash logic: manifest lookup,
  skip-if-already-done, path construction, conda activation, the new
  `--distance-cutoff` flag): executed directly (with simulated
  `SLURM_ARRAY_TASK_ID`) against real files end-to-end, not just inspected.
- **PyTorch Dataset logic**: simulated `__getitem__`'s slicing math,
  `collate_fn`'s padding/masking AND its sparse-edge re-indexing/batching
  (the shift-by-`b_idx * max_len` step, checked by hand against the
  expected batched edge list), and `to_dense_distance_matrix`'s
  symmetric-matrix reconstruction, all with numpy standing in for torch
  tensors. Verified the pickle round-trip used when DataLoader sends the
  Dataset to worker processes correctly drops and lazily re-opens the file
  handle, including the new distance datasets.

What I could **not** test directly here: real h5py/torch behavior (only
available on your cluster), and real-world throughput/memory under actual
cluster I/O — please do a small pilot (one chunk, by hand) before launching
the full array.
