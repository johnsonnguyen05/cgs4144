import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import os

# -------------------------
# Create results directory if it doesn't exist
# -------------------------
os.makedirs("results", exist_ok=True)

# -------------------------
# Load expression data
# -------------------------
df_expr = pd.read_csv("expression_with_gene_names.tsv", sep="\t")
expr = df_expr.iloc[:, 1:].T  # transpose: samples x genes
expr.index = df_expr.columns[1:]  # sample names as index

# -------------------------
# Load metadata
# -------------------------
metadata = pd.read_csv("refinebio/SRP119064/metadata_SRP119064.tsv", sep="\t")
metadata = metadata.set_index('refinebio_accession_code')

# -------------------------
# Define groups: Trem2KO vs WT
# -------------------------
groups = metadata['refinebio_subject'].reindex(expr.index)  # align to expression samples
groups = groups.fillna("Unknown")

def assign_group(x):
    x = str(x).lower()
    if "trem2ko" in x or "trem2 ko" in x:
        return "Trem2KO"
    elif "wt" in x:
        return "WT"
    else:
        return "Unknown"

groups = groups.map(assign_group)

# Check counts
print("Group counts:\n", groups.value_counts())

# Keep only Trem2KO and WT samples
valid_idx = groups[groups.isin(["Trem2KO", "WT"])].index
expr = expr.loc[valid_idx]
groups = groups.loc[valid_idx]

# -------------------------
# Log-transform and scale
# -------------------------
expr_log = np.log2(expr + 1)
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_log)

# -------------------------
# Define consistent color mapping
# -------------------------
color_map = {"Trem2KO": "red", "WT": "blue"}

# -------------------------
# PCA
# -------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expr_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=pca_result[:,0],
    y=pca_result[:,1],
    hue=groups,
    palette=color_map,
    s=100
)
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA of Expression Data (Trem2KO vs WT)")
plt.legend(title="Group", loc="best")
plt.tight_layout()
plt.savefig("results/PCA_plot.png")
plt.show()

# -------------------------
# t-SNE
# -------------------------
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(expr_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=tsne_result[:,0],
    y=tsne_result[:,1],
    hue=groups,
    palette=color_map,
    s=100
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of Expression Data (Trem2KO vs WT)")
plt.legend(title="Group", loc="best")
plt.tight_layout()
plt.savefig("results/tSNE_plot.png")
plt.show()

# -------------------------
# UMAP
# -------------------------
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(expr_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=umap_result[:,0],
    y=umap_result[:,1],
    hue=groups,
    palette=color_map,
    s=100
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Expression Data (Trem2KO vs WT)")
plt.legend(title="Group", loc="best")
plt.tight_layout()
plt.savefig("results/UMAP_plot.png")
plt.show()
