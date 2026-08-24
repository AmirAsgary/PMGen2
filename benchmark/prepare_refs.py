"""Convert benchmark reference PDBs into the PMGen teacher convention.

The reference PDBs are single-chain crystal structures (chain C, contiguous
numbering, MHC followed by the peptide).  ``src/pdb/parse.py`` expects the PMGen
AlphaFold layout instead: chain "A", MHC numbered 1..n_mhc and the peptide
restarting at n_mhc+201 so the >=150 jump marks the chain break.

Rewriting the file (rather than patching the parser) means the SAME file is read
by parse_example and by openfold's from_pdb_string, so the residue ordering used
for the model input and for the RMSD reference cannot drift apart.

Only ATOM records are kept, altloc is collapsed to the first-listed conformer,
and residues are emitted in file order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PEP_OFFSET = 200          # matches the PMGen teacher PDBs (185 -> 386)


def convert(src: Path, dst: Path, n_mhc: int, n_pep: int) -> int:
    keep, cur, ridx = [], None, -1
    for l in src.read_text().splitlines():
        if not l.startswith("ATOM"):
            continue
        alt = l[16]
        if alt not in (" ", "A"):
            continue                       # first-listed conformer only
        key = l[22:27]
        if key != cur:
            cur = key
            ridx += 1
        keep.append((ridx, l))
    n_res = ridx + 1
    if n_res != n_mhc + n_pep:
        raise ValueError(f"{src.name}: {n_res} residues, expected {n_mhc}+{n_pep}")

    out = []
    for i, l in keep:
        num = (i + 1) if i < n_mhc else (n_mhc + PEP_OFFSET + (i - n_mhc) + 1)
        out.append(f"{l[:16]} {l[17:21]}A{num:4d} {l[27:]}")
    out.append("TER")
    out.append("END")
    dst.write_text("\n".join(out) + "\n")
    return n_res


def main() -> None:
    bm = pd.read_csv(HERE / "pRMSD_benchmark.tsv", sep="\t")
    src_dir = HERE / "reference_pdbs"
    dst_dir = HERE / "converted_pdbs"
    dst_dir.mkdir(exist_ok=True)
    n = 0
    for _, r in bm.iterrows():
        convert(src_dir / f"{r.PDB}.pdb", dst_dir / f"{r.PDB}.pdb",
                len(r.mhc_seq), len(r.peptide))
        n += 1
    print(f"converted {n} reference PDBs -> {dst_dir}")


if __name__ == "__main__":
    sys.exit(main())
