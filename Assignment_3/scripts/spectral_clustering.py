#!/usr/bin/env python3
"""Run Spectral Clustering on top-5000 gene expression data.

Input: Assignment_3/expression_data_top5000.tsv (genes x samples)
Outputs:
 - Assignment_3/results/spectral_labels.tsv (sample, cluster)
 - Assignment_3/results/spectral_pca.png (2D PCA scatter colored by cluster)

Usage: python spectral_clustering.py [--input PATH] [--k K] [--output-dir DIR]
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def main(argv):
    p = argparse.ArgumentParser(description="Spectral clustering on expression data (top 5000 genes)")
    p.add_argument("--input", "-i", default="Assignment_3/expression_data_top5000.tsv", help="Input TSV (genes x samples)")
    p.add_argument("--k", "-k", type=int, default=6, help="Number of clusters (default: 6)")
    p.add_argument("--output-dir", "-o", default="Assignment_3/results", help="Output directory")
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading expression matrix from {args.input}")
    df = pd.read_csv(args.input, sep="\t", header=0)

    # Expect genes x samples: first column is gene id, remaining are sample columns
    sample_cols = df.columns[1:]
    expr = df[sample_cols].T  # now samples x genes
    expr.index = sample_cols

    # Log-transform if values look like counts (heuristic: min >= 0 and max > 100)
    if (expr.values.min() >= 0) and (expr.values.max() > 100):
        print("Applying log2(x+1) transform")
        expr = np.log2(expr + 1)

    # Standardize features
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr)

    print(f"Running Spectral Clustering with k={args.k}")
    spectral = SpectralClustering(
        n_clusters=args.k,
        affinity='nearest_neighbors',  # often better for biological data
        assign_labels='kmeans',
        random_state=42,
        n_neighbors=10
    )
    labels = spectral.fit_predict(expr_scaled)

    # Save labels
    out_labels = os.path.join(args.output_dir, "spectral_labels.tsv")
    pd.DataFrame({"sample": expr.index, "cluster": labels}).to_csv(out_labels, sep="\t", index=False)
    print(f"Wrote cluster labels to {out_labels}")

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(expr_scaled)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=labels, palette="tab10", s=80)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"Spectral Clustering (k={args.k}) on top-5000 genes")
    plt.legend(title="cluster", loc="best")
    out_plot = os.path.join(args.output_dir, "spectral_pca.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    print(f"Wrote PCA plot to {out_plot}")

    # Print cluster counts
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster counts:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")


if __name__ == "__main__":
    main(sys.argv[1:])
