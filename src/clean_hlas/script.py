import pandas as pd
import os
import utils
from collections import Counter
import subprocess

df = pd.read_csv("../../data/raw/mhc1_encodings.csv")
df["allele"] = df["key"].apply(utils.group_allele)
os.makedirs("../../analysis/raw_data_exploration/", exist_ok=True)
utils.plot_allele_composition_bar(df, "../../analysis/raw_data_exploration/hla_composition.png")
os.makedirs("../../data/raw/alleles_fasta", exist_ok=True)
utils.export_alleles_to_fasta(df, "../../data/raw/alleles_fasta")
cmd = [
    "mmseqs", "easy-cluster",
    "../../data/raw/alleles_fasta/all_hla.fa",
    "../../data/raw/alleles_clusters_all/cluster",
    "../../data/raw/alleles_clusters_all/tmp",
    "--min-seq-id", "0.85",
    "-c", "0.85",
    "--cov-mode", "1"
]
subprocess.run(cmd, check=True)