import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("expression_data.tsv", sep="\t")

# extract expression values
expr = df.iloc[:, 1:].astype(float)

# size
num_genes, num_samples = expr.shape
print(f"Matrix size: {num_genes} genes × {num_samples} samples")

# log2(x+1)
expr_log = np.log2(expr + 1)

# per-gene median expression
gene_medians = expr_log.median(axis=1)

# Plot density
plt.figure(figsize=(8,6))
gene_medians.plot(kind="density")
plt.xlabel("Per-gene median expression (log2 scale)")
plt.title("Density of per-gene median expression")
plt.savefig("plots/density_gene_plot.png")
plt.show()

print("Summary of per-gene median expression (log2 scale):")
print(gene_medians.describe())
