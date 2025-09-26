import pandas as pd
import gseapy as gp
import numpy as np

# Load top DEGs
de_df = pd.read_csv("auxillary_scripts/top50_DEGs.tsv", sep="\t")  # columns: gene, log2FoldChange, padj

# Make sure columns exist
assert "gene" in de_df.columns
assert "log2FC" in de_df.columns
assert "padj" in de_df.columns

# Prepare ranked gene list
# Use a signed metric: log2FC * -log10(padj)
de_df = de_df.dropna(subset=["log2FC", "padj"])
de_df["rank_metric"] = de_df["log2FC"] * -np.log10(de_df["padj"])

de_df["gene"] = de_df["gene"].str.upper()

# Create Series for GSEA: index=gene, value=rank_metric
ranked_genes = pd.Series(
    data=de_df["rank_metric"].values,
    index=de_df["gene"]
).sort_values(ascending=False)

# GSEA using Gene Ontology
gsea_res = gp.prerank(
    rnk=ranked_genes,
#     gene_sets=[
#         "KOMP2_Mouse_Phenotypes_2022",
#         "MGI_Mammalian_Phenotype_Level_4_2024",
#     ],
    gene_sets=["GO_Biological_Process_2021"],  
    outdir=None,
    permutation_num=100,  # increase for more robust results
    min_size=3,
    max_size=1000
)

# Export results
gsea_res.res2d.to_csv("results/prerank_gene.tsv", sep="\t")

print("GO enrichment analysis complete. Results saved to 'results/prerank_gene.tsv'.")
