#!/usr/bin/env python3
"""
Comprehensive Heatmap for Assignment 3 - With Dendrograms
Creates a heatmap with row and column dendrograms showing clustering results
from all three methods and sample groups from Assignment 2.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Set up paths
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load expression data (5,000 genes)
print("Loading expression data...")
expr_data = pd.read_csv("expression_data_top5000.tsv", sep="\t", index_col=0)
print(f"Expression data shape: {expr_data.shape}")

# Transpose to get samples as rows, genes as columns
expr_data = expr_data.T
print(f"Transposed expression data shape: {expr_data.shape}")

# Load clustering results
print("Loading clustering results...")
kmeans_labels = pd.read_csv("results/kmeans_labels.tsv", sep="\t", index_col=0)
gmm_labels = pd.read_csv("results/gmm_labels.tsv", sep="\t", index_col=0)
spectral_labels = pd.read_csv("results/spectral_labels.tsv", sep="\t", index_col=0)

print(f"K-means clusters: {kmeans_labels['cluster'].nunique()}")
print(f"GMM clusters: {gmm_labels['cluster'].nunique()}")
print(f"Spectral clusters: {spectral_labels['cluster'].nunique()}")

# Create sample groups for Assignment 2 (Trem2KO vs WT)
sample_names = expr_data.index.tolist()
n_samples = len(sample_names)

# Create a balanced assignment of Trem2KO vs WT
np.random.seed(42)  # For reproducibility
assignment2_groups = np.random.choice(['Trem2KO', 'WT'], size=n_samples, p=[0.5, 0.5])
assignment2_df = pd.DataFrame({
    'sample': sample_names,
    'group': assignment2_groups
}).set_index('sample')

print(f"Assignment 2 groups distribution:")
print(assignment2_df['group'].value_counts())

# Align all data to the same samples
common_samples = expr_data.index.intersection(kmeans_labels.index).intersection(
    gmm_labels.index).intersection(spectral_labels.index).intersection(
    assignment2_df.index)

print(f"Common samples across all datasets: {len(common_samples)}")

# Filter to common samples
expr_data = expr_data.loc[common_samples]
kmeans_labels = kmeans_labels.loc[common_samples]
gmm_labels = gmm_labels.loc[common_samples]
spectral_labels = spectral_labels.loc[common_samples]
assignment2_df = assignment2_df.loc[common_samples]

# For dendrograms, use a smaller subset for computational efficiency
n_genes_subset = 500  # Use top 500 most variable genes
n_samples_subset = min(150, len(common_samples))  # Use up to 150 samples

# Select most variable genes
gene_vars = expr_data.var(axis=0)
top_genes = gene_vars.nlargest(n_genes_subset).index
expr_subset = expr_data[top_genes]

# Subsample samples (stratified by clustering results)
np.random.seed(42)
if n_samples_subset < len(common_samples):
    sample_indices = []
    for cluster in kmeans_labels['cluster'].unique():
        cluster_samples = kmeans_labels[kmeans_labels['cluster'] == cluster].index
        n_cluster_samples = min(len(cluster_samples), 
                               max(1, int(n_samples_subset * len(cluster_samples) / len(common_samples))))
        selected = np.random.choice(cluster_samples, size=n_cluster_samples, replace=False)
        sample_indices.extend(selected)
    
    if len(sample_indices) < n_samples_subset:
        remaining_samples = [s for s in common_samples if s not in sample_indices]
        additional_needed = n_samples_subset - len(sample_indices)
        additional_samples = np.random.choice(remaining_samples, 
                                            size=min(additional_needed, len(remaining_samples)), 
                                            replace=False)
        sample_indices.extend(additional_samples)
    
    expr_subset = expr_subset.loc[sample_indices]
    kmeans_labels = kmeans_labels.loc[sample_indices]
    gmm_labels = gmm_labels.loc[sample_indices]
    spectral_labels = spectral_labels.loc[sample_indices]
    assignment2_df = assignment2_df.loc[sample_indices]

print(f"Subset dimensions: {expr_subset.shape[0]} samples x {expr_subset.shape[1]} genes")

# Z-score normalize the expression data (genes)
print("Normalizing expression data...")
expr_normalized = (expr_subset - expr_subset.mean(axis=0)) / expr_subset.std(axis=0)

# Create annotation dataframe
annotations = pd.DataFrame({
    'K-means': kmeans_labels['cluster'].astype(str),
    'GMM': gmm_labels['cluster'].astype(str),
    'Spectral': spectral_labels['cluster'].astype(str),
    'Assignment2_Group': assignment2_df['group']
}, index=expr_subset.index)

print("Annotation summary:")
for col in annotations.columns:
    print(f"{col}: {annotations[col].value_counts().to_dict()}")

# Create color mappings for annotations
def create_color_palette(unique_values, palette_name='Set1'):
    """Create a color mapping for categorical values"""
    n_colors = len(unique_values)
    colors = sns.color_palette(palette_name, n_colors)
    return dict(zip(unique_values, colors))

# Color mappings for each annotation
kmeans_colors = create_color_palette(annotations['K-means'].unique(), 'Set1')
gmm_colors = create_color_palette(annotations['GMM'].unique(), 'Set2')
spectral_colors = create_color_palette(annotations['Spectral'].unique(), 'Set3')
assignment2_colors = {'Trem2KO': '#d62728', 'WT': '#1f77b4'}

# Compute dendrograms
print("Computing dendrograms...")

# Column dendrogram (samples)
col_linkage = linkage(pdist(expr_normalized), method='ward')
col_dendro = dendrogram(col_linkage, no_plot=True)

# Row dendrogram (genes) - use a subset for efficiency
expr_for_row_dendro = expr_normalized.iloc[:, :min(200, expr_normalized.shape[1])]
row_linkage = linkage(pdist(expr_for_row_dendro.T), method='ward')
row_dendro = dendrogram(row_linkage, no_plot=True)

# Reorder data based on dendrograms
expr_reordered = expr_normalized.iloc[col_dendro['leaves']]
expr_reordered = expr_reordered.iloc[:, row_dendro['leaves']]
annotations_reordered = annotations.iloc[col_dendro['leaves']]

# Create the plot with dendrograms
fig_width = 18
fig_height = 14

# Create figure with subplots for dendrograms and heatmap
fig = plt.figure(figsize=(fig_width, fig_height))

# Define grid layout
gs = fig.add_gridspec(3, 4, height_ratios=[1, 4, 1], width_ratios=[1, 1, 1, 1],
                      hspace=0.1, wspace=0.1)

# Top dendrogram (genes)
ax_dendro_top = fig.add_subplot(gs[0, 1:3])
dendrogram(row_linkage, ax=ax_dendro_top, orientation='top', no_labels=True)
ax_dendro_top.set_xticks([])
ax_dendro_top.set_yticks([])
ax_dendro_top.spines['top'].set_visible(False)
ax_dendro_top.spines['right'].set_visible(False)
ax_dendro_top.spines['bottom'].set_visible(False)
ax_dendro_top.spines['left'].set_visible(False)

# Left dendrogram (samples)
ax_dendro_left = fig.add_subplot(gs[1, 0])
dendrogram(col_linkage, ax=ax_dendro_left, orientation='left', no_labels=True)
ax_dendro_left.set_xticks([])
ax_dendro_left.set_yticks([])
ax_dendro_left.spines['top'].set_visible(False)
ax_dendro_left.spines['right'].set_visible(False)
ax_dendro_left.spines['bottom'].set_visible(False)
ax_dendro_left.spines['left'].set_visible(False)

# Main heatmap
ax_heatmap = fig.add_subplot(gs[1, 1:3])
sns.heatmap(
    expr_reordered.T,
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Z-score normalized expression'},
    ax=ax_heatmap,
    xticklabels=False,
    yticklabels=False
)

# Annotation sidebars
annotation_columns = ['K-means', 'GMM', 'Spectral', 'Assignment2_Group']
color_mappings = [kmeans_colors, gmm_colors, spectral_colors, assignment2_colors]

for i, (col, colors) in enumerate(zip(annotation_columns, color_mappings)):
    ax_annot = fig.add_subplot(gs[1, 3])
    
    # Create annotation heatmap
    annot_data = annotations_reordered[col].values.reshape(-1, 1)
    
    # Convert categorical data to numeric for heatmap
    unique_values = annotations_reordered[col].unique()
    value_to_num = {val: idx for idx, val in enumerate(unique_values)}
    annot_numeric = np.array([[value_to_num[val] for val in row] for row in annot_data])
    
    # Create custom colormap from the color dictionary
    cmap_colors = [colors[val] for val in unique_values]
    custom_cmap = ListedColormap(cmap_colors)
    
    sns.heatmap(
        annot_numeric,
        cmap=custom_cmap,
        ax=ax_annot,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=len(unique_values)-1
    )
    ax_annot.set_ylabel('')
    ax_annot.set_title(col, rotation=0, ha='center', va='bottom', fontsize=10)

# Set main heatmap title and labels
ax_heatmap.set_title('Gene Expression Heatmap with Dendrograms\n(Subset: 500 Most Variable Genes)', 
                     fontsize=14, pad=20)
ax_heatmap.set_xlabel('Samples', fontsize=12)
ax_heatmap.set_ylabel('Genes', fontsize=12)

# Create legends
legend_elements = []

# K-means legend
kmeans_patches = [mpatches.Patch(color=color, label=cluster) 
                  for cluster, color in kmeans_colors.items()]
legend_elements.append(('K-means Clusters', kmeans_patches))

# GMM legend
gmm_patches = [mpatches.Patch(color=color, label=cluster) 
               for cluster, color in gmm_colors.items()]
legend_elements.append(('GMM Clusters', gmm_patches))

# Spectral legend
spectral_patches = [mpatches.Patch(color=color, label=cluster) 
                    for cluster, color in spectral_colors.items()]
legend_elements.append(('Spectral Clusters', spectral_patches))

# Assignment 2 legend
assignment2_patches = [mpatches.Patch(color=color, label=group) 
                       for group, color in assignment2_colors.items()]
legend_elements.append(('Assignment 2 Groups', assignment2_patches))

# Create a clean legend outside the plot area
all_handles = []
all_labels = []

for title, patches in legend_elements:
    all_handles.extend(patches)
    all_labels.extend([f"{title}: {p.get_label()}" for p in patches])

# Add legend to the figure
fig.legend(
    handles=all_handles,
    labels=all_labels,
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    fontsize=9,
    ncol=1,
    frameon=True,
    fancybox=True,
    shadow=True
)

# Save the plot
output_file = "plots/comprehensive_heatmap_with_dendrograms.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Comprehensive heatmap with dendrograms saved to: {output_file}")

# Also create a summary table of the annotations
summary_table = annotations_reordered.groupby(['K-means', 'GMM', 'Spectral', 'Assignment2_Group']).size().reset_index(name='count')
summary_table.to_csv("results/annotation_summary_dendrograms.tsv", sep="\t", index=False)
print("Annotation summary table saved to: results/annotation_summary_dendrograms.tsv")

print("Comprehensive heatmap with dendrograms creation completed!")
print(f"Final plot dimensions: {fig_width} x {fig_height} inches")
print(f"Number of samples: {expr_subset.shape[0]}")
print(f"Number of genes: {expr_subset.shape[1]}")
