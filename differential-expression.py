import os
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from bioinfokit import visuz
import bioinfokit
from statsmodels.stats.multitest import multipletests
print(f"Running bioinfokit version: {bioinfokit.__version__}")

print("Step 4.1: Libraries imported successfully.")


data_file = "expression_with_gene_names.tsv"
metadata_file = "refinebio/SRP119064/metadata_SRP119064.tsv"

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

# 1. Explicitly run the statistical test.
stat_res.run_wald_test()
# 2. Manually build the results DataFrame from the available attributes.
res_df = pd.DataFrame(
    {
        "baseMean": stat_res.base_mean,
        # --- START OF FINAL CORRECTION ---
        # Use .iloc to select the first column from the LFC DataFrame
        "log2FoldChange": stat_res.LFC.iloc[:, 0],
        # --- END OF FINAL CORRECTION ---
        "lfcSE": stat_res.SE,
        "stat": stat_res.statistics,
        "pvalue": stat_res.p_values,
    }
)
# Calculate adjusted p-values, handling NaN values
# FIX: Use 'fdr_bh' instead of 'BH' for statsmodels
not_na = res_df['pvalue'].notna()
res_df.loc[not_na, 'padj'] = multipletests(res_df.loc[not_na, 'pvalue'], method='fdr_bh')[1]

print("    Analysis complete.")

# Save results to TSV
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

deseq_df_sorted = res_df.sort_values('padj').reset_index().rename(columns={'index': 'Gene'})
output_path = os.path.join(results_dir, "SRP119064_diff_expr_results.tsv")
deseq_df_sorted.to_csv(output_path, sep='\t', index=False)

print(f"Step 4.7: Full results table saved to: {output_path}")

# Create a volcano plot
res_df_bioinfokit = res_df.reset_index().rename(columns={'index': 'Gene'})

# Remove rows with NaN values in required columns for plotting
res_df_clean = res_df_bioinfokit.dropna(subset=['log2FoldChange', 'padj']).copy()

print("Step 4.8: Generating volcano plot with bioinfokit...")
print(f"    Using {len(res_df_clean)} genes for plotting (after removing NaN values)")

# Use the working method from bioinfokit 2.1.4
visuz.GeneExpression.volcano(
    df=res_df_clean, 
    lfc='log2FoldChange', 
    pv='padj',
    lfc_thr=(1.0, 1.0),
    pv_thr=(0.05, 0.05),
    show=False,
    figtype='png',
    axtickfontsize=12,
    axlabelfontsize=12,
    dim=(8, 6),
    r=300
)

# Move the volcano plot to the results directory
import shutil
volcano_source = "volcano.png"
volcano_dest = os.path.join(results_dir, "volcano.png")
if os.path.exists(volcano_source):
    shutil.move(volcano_source, volcano_dest)
    print(f"    Volcano plot saved to: {volcano_dest}")
else:
    print("    Warning: volcano.png not found in current directory")

print("    Volcano plot created successfully with GeneExpression.volcano()")
print("\nScript finished successfully! ✅")