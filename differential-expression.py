import os
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from bioinfokit import visuz
import bioinfokit
from statsmodels.stats.multitest import multipletests
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print(f"Running bioinfokit version: {bioinfokit.__version__}")

print("Step 4.1: Libraries imported successfully.")


data_file = "expression_with_gene_names.tsv"
metadata_file = "refinebio/SRP119064/metadata_SRP119064.tsv"

# load data and metadata
expression_df = pd.read_csv(data_file, sep='\t', index_col=0)
metadata_df = pd.read_csv(metadata_file, sep='\t', index_col=0)

# Force the index (sample names) to be strings
metadata_df.index = metadata_df.index.astype(str)
expression_df.columns = expression_df.columns.astype(str)

# Ensure expression columns match metadata index order
expression_df = expression_df[metadata_df.index]

if not all(expression_df.columns == metadata_df.index):
    raise ValueError("Mismatch between expression data columns and metadata index.")
print(f"Step 4.2: Loaded {expression_df.shape[0]} genes and {expression_df.shape[1]} samples.")


# metadata preparation
# The condition info is in the 'refinebio_subject' column.
conditions = []
for subject_info in metadata_df['refinebio_subject']:
    if isinstance(subject_info, str):
        parts = [part.strip() for part in subject_info.split(',')]
        if "trem2ko" in parts:
            conditions.append("trem2ko")
        elif "wt" in parts:
            conditions.append("reference")
        else:
            conditions.append(None)
    else:
        conditions.append(None)
metadata_df['mutation_status'] = conditions

print("Step 4.3: Created 'mutation_status' column in metadata.")

# Filter out samples with 'None' condition
metadata_filtered = metadata_df.dropna(subset=['mutation_status'])
expression_df_filtered = expression_df[metadata_filtered.index]
print(f"    Filtered data to keep {len(metadata_filtered)} samples with valid conditions.")

# Define a minimum counts cutoff
min_counts = 10
filtered_expression_df = expression_df_filtered[expression_df_filtered.sum(axis=1) >= min_counts]
print("Step 4.4: Filtered low-count genes.")

# DESeq2 analysis
# Create a DeseqDataSet and run analysis
counts_df_int = filtered_expression_df.round().astype(int)
counts_transposed = counts_df_int.T

dds = DeseqDataSet(
    counts=counts_transposed,
    metadata=metadata_filtered,
    design="~ mutation_status"
)

print("Step 4.5-4.6: Running DESeq2 analysis...")
dds.deseq2()

# Compare the 'trem2ko' group against the 'reference' (wt) group
stat_res = DeseqStats(dds, contrast=("mutation_status", "trem2ko", "reference"))

# Run the statistical test and build the results DataFrame
stat_res.run_wald_test()
res_df = pd.DataFrame(
    {
        "baseMean": stat_res.base_mean,
        "log2FoldChange": stat_res.LFC.iloc[:, 0],
        "lfcSE": stat_res.SE,
        "stat": stat_res.statistics,
        "pvalue": stat_res.p_values,
    }
)
# Calculate adjusted p-values, handling NaN values
not_na = res_df['pvalue'].notna()
res_df.loc[not_na, 'padj'] = multipletests(res_df.loc[not_na, 'pvalue'], method='fdr_bh')[1]

print("    Analysis complete.")

# save results
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

deseq_df_sorted = res_df.sort_values('padj').reset_index().rename(columns={'index': 'Gene'})
output_path = os.path.join(results_dir, "SRP119064_diff_expr_results.tsv")
deseq_df_sorted.to_csv(output_path, sep='\t', index=False)

print(f"Step 4.7: Full results table saved to: {output_path}")

# generate top 50 genes data and visual table
# Save the top 50 differentially expressed genes to a new TSV file
top_50_genes = deseq_df_sorted.head(50)
top_50_output_path = os.path.join(results_dir, "top50.tsv")
top_50_genes.to_csv(top_50_output_path, sep='\t', index=False)
print(f"    Top 50 differentially expressed genes (data) saved to: {top_50_output_path}")

# generate heatmap for top 50 genes
print("    Generating visual table (heatmap) for top 50 genes...")
# Get the gene names
top_50_gene_names = top_50_genes['Gene'].tolist()

# Get normalized counts from dds object.
# The counts were transposed for DeseqDataSet, so normalized counts have samples as rows and genes as columns.
normalized_counts = pd.DataFrame(dds.layers['normed_counts'], index=counts_transposed.index, columns=counts_transposed.columns)

# Filter for top 50 genes and log2 transform (add 1 to avoid log(0))
top_50_norm_counts = normalized_counts[top_50_gene_names]
log_transformed_counts = np.log2(top_50_norm_counts + 1)

# Transpose so that genes are on the y-axis and samples are on the x-axis
data_for_heatmap = log_transformed_counts.T

# Create color annotations for samples based on mutation status
# This helps in visualizing which samples belong to which group
unique_conditions = metadata_filtered['mutation_status'].unique()
palette = sns.color_palette("hsv", len(unique_conditions))
color_map = dict(zip(unique_conditions, palette))
sample_colors = metadata_filtered['mutation_status'].map(color_map)

# Generate the clustermap. It clusters both genes and samples.
# z_score=0 normalizes each gene's expression across samples to see relative changes.
g = sns.clustermap(data_for_heatmap,
                   z_score=0,
                   cmap='vlag',
                   col_colors=sample_colors,
                   figsize=(12, 18),
                   yticklabels=True)

# Adjust labels for better readability
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=90)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=8)
g.fig.suptitle('Top 50 Differentially Expressed Genes', fontsize=16, y=1.02)

# Save the heatmap to a file
heatmap_path = os.path.join(results_dir, "top50_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close() # Important to close the figure to free memory

print(f"    Visual table (heatmap) saved to: {heatmap_path}")

# generate volcano plot
res_df_bioinfokit = res_df.reset_index().rename(columns={'index': 'Gene'})

# Remove rows with NaN values in required columns for plotting
res_df_clean = res_df_bioinfokit.dropna(subset=['log2FoldChange', 'padj']).copy()

print("Step 4.8: Generating volcano plot (custom matplotlib)...")
print(f"    Using {len(res_df_clean)} genes for plotting (after removing NaN values)")

# Define thresholds
lfc_thresh = 1.0
padj_thresh = 0.05

# Protect against padj == 0
padj = res_df_clean['padj'].replace(0, 1e-300)
neg_log10_padj = -np.log10(padj)

# Categorize genes
def category(row):
    if row['padj'] < padj_thresh and row['log2FoldChange'] >= lfc_thresh:
        return 'Up'
    elif row['padj'] < padj_thresh and row['log2FoldChange'] <= -lfc_thresh:
        return 'Down'
    else:
        return 'NotSig'

cats = res_df_clean.apply(category, axis=1)
res_df_clean['category'] = cats
res_df_clean['neg_log10_padj'] = neg_log10_padj

# Colors for categories
colors = {'Up':'red', 'Down':'blue', 'NotSig':'grey'}

fig, ax = plt.subplots(figsize=(8,6))

# Plot non-significant points first (so significant are on top)
for cat in ['NotSig','Down','Up']:
    subset = res_df_clean[res_df_clean['category'] == cat]
    ax.scatter(subset['log2FoldChange'], subset['neg_log10_padj'],
               c=colors[cat], alpha=0.6, s=10, label=cat)

# Add threshold lines
ax.axvline(x=lfc_thresh, color='black', linestyle='--', linewidth=0.8)
ax.axvline(x=-lfc_thresh, color='black', linestyle='--', linewidth=0.8)
ax.axhline(y=-np.log10(padj_thresh), color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel('log2 Fold Change', fontsize=12)
ax.set_ylabel('-log10(adjusted p-value)', fontsize=12)
ax.set_title('Volcano plot: Differential expression', fontsize=14)

# Legend
legend = ax.legend(title='Key', frameon=True, fontsize=10)
plt.setp(legend.get_title(), fontsize=11)

# Save directly to results directory
volcano_dest = os.path.join(results_dir, 'volcano.png')
plt.tight_layout()
plt.savefig(volcano_dest, dpi=300)
plt.close()
print(f"    Volcano plot saved to: {volcano_dest}")
print("    Volcano plot created successfully (custom)")
print("\nScript finished successfully!")

