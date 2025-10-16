#!/usr/bin/env python3
"""Run clustering with different numbers of genes and save results."""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_expression_data(input_file):
    """Load and preprocess expression data."""
    print(f"Loading expression data from {input_file}")
    df = pd.read_csv(input_file, sep="\t", header=0)
    
    # Expect genes x samples: first column is gene id, remaining are sample columns
    sample_cols = df.columns[1:]
    expr = df[sample_cols].T  # now samples x genes
    expr.index = sample_cols
    
    # Log-transform if values look like counts (heuristic: min >= 0 and max > 100)
    if (expr.values.min() >= 0) and (expr.values.max() > 100):
        print("Applying log2(x+1) transform")
        expr = np.log2(expr + 1)
    
    return expr

def subset_genes(expr, n_genes):
    """Subset to top n_genes by variance."""
    if n_genes >= expr.shape[1]:
        return expr
    
    # Calculate variance for each gene
    gene_vars = expr.var(axis=0)
    # Get top n_genes by variance
    top_genes_idx = gene_vars.nlargest(n_genes).index
    return expr[top_genes_idx]

def run_clustering(expr, method, k, random_state=42):
    """Run clustering with specified method and parameters."""
    # Standardize features
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(expr_scaled)
    elif method == 'gmm':
        clusterer = GaussianMixture(n_components=k, random_state=random_state)
        clusterer.fit(expr_scaled)
        labels = clusterer.predict(expr_scaled)
    elif method == 'spectral':
        clusterer = SpectralClustering(
            n_clusters=k,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=random_state,
            n_neighbors=10
        )
        labels = clusterer.fit_predict(expr_scaled)
    
    return labels, expr_scaled

def analyze_cluster_counts(labels, method_name):
    """Analyze and print cluster counts."""
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{method_name} cluster counts:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")
    return dict(zip(unique, counts))

def main():
    # Set up paths
    input_file = "Assignment_3/expression_data_top5000.tsv"
    results_dir = "Assignment_3/results"
    varied_genes_dir = "Assignment_3/varied_genes_results"
    
    os.makedirs(varied_genes_dir, exist_ok=True)
    
    # Load data
    expr_full = load_expression_data(input_file)
    print(f"Full dataset shape: {expr_full.shape}")
    
    # Run clustering with different numbers of genes
    gene_numbers = [10, 100, 1000, 10000]
    methods = ['kmeans', 'gmm', 'spectral']
    k_default = 6
    
    print("\n" + "="*60)
    print("RUNNING CLUSTERING WITH DIFFERENT NUMBERS OF GENES")
    print("="*60)
    
    for method in methods:
        print(f"\n--- {method.upper()} ---")

        # Create subdirectory for each method
        method_dir = os.path.join(varied_genes_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        for n_genes in gene_numbers:
            print(f"\nRunning {method} with {n_genes} genes")
            expr_subset = subset_genes(expr_full, n_genes)
            print(f"Subset shape: {expr_subset.shape}")
            
            labels, expr_scaled = run_clustering(expr_subset, method, k_default)
            
            # Save labels
            out_labels = os.path.join(method_dir, f"genes{n_genes}_labels.tsv")
            pd.DataFrame({"sample": expr_subset.index, "cluster": labels}).to_csv(out_labels, sep="\t", index=False)
            print(f"Saved labels to {out_labels}")
            
            # Analyze cluster counts
            cluster_counts = analyze_cluster_counts(labels, f"{method.upper()} {n_genes} genes")
            
            # Create PCA plot
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(expr_scaled)
            
            plt.figure(figsize=(7, 6))
            sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=labels, palette="tab10", s=80)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.title(f"{method.upper()} ({n_genes} genes)")
            plt.legend(title="cluster", loc="best")
            
            out_plot = os.path.join(method_dir, f"genes{n_genes}_pca.png")
            plt.tight_layout()
            plt.savefig(out_plot, dpi=200)
            plt.close()
            print(f"Saved PCA plot to {out_plot}")


if __name__ == "__main__":
    main()
