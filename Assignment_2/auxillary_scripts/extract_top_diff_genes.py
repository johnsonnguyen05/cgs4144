import pandas as pd

# Load DE results
df = pd.read_csv("results/differential_expression_data.tsv", sep="\t")

# Option 1: top 50 by adjusted p-value (most significant)
top50_padj = df.sort_values("padj").head(50)

# Option 2: top 50 by absolute log2 fold change (most strongly changed)
top50_log2fc = df.reindex(df["log2FC"].abs().sort_values(ascending=False).index).head(50)

# Save to TSV
top50_padj.to_csv("auxillary_scripts/top50_DEGs.tsv", sep="\t", index=False)

print("Top 50 DEGs saved!")
