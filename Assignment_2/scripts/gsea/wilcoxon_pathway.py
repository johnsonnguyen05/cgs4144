import pandas as pd
import gseapy as gp
import numpy as np
from scipy.stats import ranksums

# Load top DEGs
de_df = pd.read_csv("auxillary_scripts/top50_DEGs.tsv", sep="\t")  # columns: gene, log2FC, padj

# Ensure columns exist
assert "gene" in de_df.columns
assert "log2FC" in de_df.columns
assert "padj" in de_df.columns

# Prepare ranked gene list
de_df = de_df.dropna(subset=["log2FC", "padj"])
de_df["rank_metric"] = de_df["log2FC"] * -np.log10(de_df["padj"])
de_df["gene"] = de_df["gene"].str.upper()

ranked_genes = pd.Series(
    data=de_df["rank_metric"].values,
    index=de_df["gene"]
).sort_values(ascending=False)

# Perform GSEA using a pathway ontology
# We'll use Reactome pathways as an example
gsea_res = gp.prerank(
    rnk=ranked_genes,
    gene_sets="Reactome_2022",   # pathway ontology
    outdir=None,
    permutation_num=100,         # increase for robustness
    min_size=3,
    max_size=500,
    method='wilcoxon'            # Wilcoxon rank-sum test
)

# Export results
gsea_res.res2d.to_csv("results/wilcoxon_pathway.tsv", sep="\t")

print("Reactome pathway enrichment using Wilcoxon test complete.")
