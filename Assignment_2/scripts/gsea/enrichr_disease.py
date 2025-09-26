import pandas as pd
import gseapy as gp

#print(gp.get_library_name(organism='Mouse'))

# Load DE results and metadata
de_df = pd.read_csv("auxillary_scripts/top50_DEGs.tsv", sep="\t")  # columns: gene, log_fold_change, fdr, etc.
metadata = pd.read_csv("SRP119064/metadata_SRP119064.tsv", sep="\t")  # contains sample information

# Prepare ranked gene list
# GSEA typically uses a ranking metric like log2 fold change
# We rank genes by log_fold_change (descending)
ranked_genes = de_df.set_index("gene")["log2FC"].sort_values(ascending=False)

# Perform gene set enrichment analysis using Disease Ontology
# Using Enrichr libraries related to diseases
enr = gp.enrichr(
    gene_list=ranked_genes.index.tolist(),
    gene_sets=[
        "KOMP2_Mouse_Phenotypes_2022",
        "MGI_Mammalian_Phenotype_Level_4_2024",
    ],
    organism='Mouse',
    outdir=None  # skip plots
)

# Combine enrichment results and save to TSV
do_results = pd.concat([enr.results], ignore_index=True)
do_results.to_csv("results/enrichr_disease.tsv", sep="\t", index=False)

print("Disease Ontology gene set enrichment complete. Results saved to 'enrichr_disease.tsv'.")
