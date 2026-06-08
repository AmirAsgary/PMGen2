import pandas as pd
import os
import matplotlib.pyplot as plt

def group_allele(key):
    """
    Map allele names into broad groups.
    Usage:
        df["allele"] = df["key"].apply(group_allele)
    """
    key = str(key)
    # Human classical MHC-I
    if key.startswith("HLA-A"):
        return "HLA-A"
    elif key.startswith("HLA-B"):
        return "HLA-B"
    elif key.startswith("HLA-C"):
        return "HLA-C"
    # Human non-classical MHC-I
    elif key.startswith("HLA-E"):
        return "HLA-E"
    elif key.startswith("HLA-F"):
        return "HLA-F"
    elif key.startswith("HLA-G"):
        return "HLA-G"
    # Rhesus macaque
    elif key.startswith("Mamu"):
        return "Mamu"
    # Swine
    elif key.startswith("SLA"):
        return "SLA"
    # Gorilla
    elif key.startswith("Gogo"):
        return "Gogo"
    # Chimpanzee
    elif key.startswith("Patr"):
        return "Patr"
    # Cow
    elif key.startswith(("BOLA", "BoLA", "BolA")):
        return "BOLA"
    # Horse
    elif key.startswith("Eqca"):
        return "Eqca"
    # Mouse
    elif key.startswith("H-2") or key.startswith("mice"):
        return "H2"
    # Everything else
    else:
        return "Other"


def plot_allele_composition_bar(
    df,
    output_path,
    allele_column="allele",
    figsize=(10, 6),
    dpi=600,
    title="Allele Composition"
):
    """
    Save a barplot of allele composition + CSV summary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Count alleles
    counts = (
        df[allele_column]
        .value_counts()
        .sort_values(ascending=True)  # ascending for horizontal barplot
    )
    # Create summary table
    summary_df = pd.DataFrame({
        "allele": counts.index,
        "count": counts.values,
        "percentage": counts.values / counts.values.sum() * 100
    }).sort_values("count", ascending=False)
    # Save CSV
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    summary_df.to_csv(csv_path, index=False)
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(counts.index, counts.values)
    plt.xlabel("Count")
    plt.ylabel("Allele Group")
    plt.title(title)
    plt.tight_layout()
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {output_path}")
    print(f"Saved CSV to: {csv_path}")


def export_alleles_to_fasta(
    df,
    output_dir,
    seq_column="mhc_sequence",
    key_column="key",
    allele_column="allele",
    combined_name="all_hla.fa",
):
    """
    Export sequences grouped by allele into separate FASTA files
    AND a combined FASTA file.
    Output:
        output_dir/
            HLA-A.fa
            HLA-B.fa
            ...
            Other.fa
            all_hla.fa
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df.dropna(subset=[seq_column, key_column, allele_column])
    combined_path = os.path.join(output_dir, combined_name)
    with open(combined_path, "w") as combined_f:
        # Group by allele
        for allele, subdf in df.groupby(allele_column):
            fasta_path = os.path.join(output_dir, f"{allele}.fa")
            with open(fasta_path, "w") as f:
                for _, row in subdf.iterrows():
                    key = str(row[key_column])
                    seq = str(row[seq_column])
                    fasta_entry = f">{key}\n{seq}\n"
                    f.write(fasta_entry)
                    combined_f.write(fasta_entry)
            print(f"Saved: {fasta_path} ({len(subdf)} sequences)")
    print(f"Saved combined FASTA: {combined_path}")