#!/usr/bin/env python3
"""
Extract significantly differentially expressed genes and create heatmap with sample groupings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_differential_expression_results(file_path):
    """Load differential expression results."""
    print("Loading differential expression results...")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} genes")
    return df

def filter_significant_genes(df, padj_threshold=0.05, log2fc_threshold=1.0):
    """Filter significantly differentially expressed genes."""
    print("Filtering significantly differentially expressed genes...")
    
    # Filter for significant genes
    significant_up = df[(df['padj'] < padj_threshold) & (df['log2FoldChange'] > log2fc_threshold)]
    significant_down = df[(df['padj'] < padj_threshold) & (df['log2FoldChange'] < -log2fc_threshold)]
    
    significant_genes = pd.concat([significant_up, significant_down])
    
    print(f"Found {len(significant_up)} significantly upregulated genes")
    print(f"Found {len(significant_down)} significantly downregulated genes")
    print(f"Total significant genes: {len(significant_genes)}")
    
    return significant_genes

def load_expression_matrix(file_path):
    """Load expression matrix."""
    print("Loading expression matrix...")
    # Read only the first few rows to get column names
    df_sample = pd.read_csv(file_path, sep='\t', nrows=5)
    sample_names = df_sample.columns[1:].tolist()  # Skip 'Gene' column
    
    # Read the full matrix
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    print(f"Loaded expression matrix with {df.shape[0]} genes and {df.shape[1]} samples")
    return df, sample_names




def create_simple_heatmap(expression_data, significant_genes):
    """Create a simple heatmap using seaborn."""
    print("Creating simple heatmap with seaborn...")
    
    # Filter expression data for significant genes only
    sig_gene_names = significant_genes['Gene'].tolist()
    # Find genes that exist in both datasets
    available_genes = expression_data.index.intersection(sig_gene_names)
    expression_subset = expression_data.loc[available_genes]
    
    print(f"Expression data for {expression_subset.shape[0]} significant genes")
    
    # Check if we have any genes to plot
    if expression_subset.shape[0] == 0:
        print("No significant genes found in expression matrix!")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Create main expression heatmap
    sns.heatmap(expression_subset, cmap='RdBu_r', center=0, 
                ax=ax, cbar_kws={'label': 'Log2 Expression'})
    ax.set_title('Significantly Differentially Expressed Genes')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes')
    
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    print("Starting analysis of significantly differentially expressed genes...")
    
    # File paths
    diff_expr_file = "results/SRP119064_diff_expr_results.tsv"
    expression_file = "expression_with_gene_names.tsv"
    
    # Load data
    diff_expr_df = load_differential_expression_results(diff_expr_file)
    expression_df, sample_names = load_expression_matrix(expression_file)
    
    # Filter significant genes
    significant_genes = filter_significant_genes(diff_expr_df)
    
    # Save significant genes list
    significant_genes.to_csv("results/significant_genes_list.tsv", sep='\t', index=False)
    print("Saved list of significant genes to results/significant_genes_list.tsv")
    
    # Create simple heatmap
    print("Creating simple heatmap...")
    fig = create_simple_heatmap(expression_df, significant_genes)
    if fig is not None:
        plt.savefig("results/significant_genes_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved heatmap to results/significant_genes_heatmap.png")
    else:
        print("No genes available for heatmap")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
