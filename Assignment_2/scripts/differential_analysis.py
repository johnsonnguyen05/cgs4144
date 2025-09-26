#!/usr/bin/env python3
"""
diffexpr_bioinfokit.py
Differential expression (WT vs Trem2KO) using per-gene Welch's t-test,
BH multiple-testing correction, and volcano/heatmap plotting.

Outputs:
 - results/diffexp_wt_vs_trem2ko.tsv
 - plots/volcano.png
 - plots/top_genes_heatmap.png --> step 4

CLEAN THIS CODE THOROUGHLY
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# create results/plots dirs
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# -------------------------
# User-tweakable parameters
# -------------------------
EXPR_FP = "expression_data.tsv"
META_FP = "SRP119064/metadata_SRP119064.tsv"
GROUP_COL_IN_META = "refinebio_subject"   # column used to detect Trem2KO vs WT
META_INDEX_COL = "refinebio_accession_code"
GROUP_NAME_TREM2 = "Trem2KO"
GROUP_NAME_WT = "WT"
LOG_TRANSFORM = True   # do log2(x+1) transform before stats
MIN_SAMPLES_PER_GROUP = 2
TOP_N_GENES_HEATMAP = 50
# significance cutoffs for labeling
PVALUE_CUTOFF = 0.05
PADJ_CUTOFF = 0.1
LFC_CUTOFF = 0.25  # abs(log2FC) threshold for "biological significance"

# -------------------------
# Helper: group normalizer
# -------------------------
def assign_group(x):
    x = str(x).lower()
    if "trem2ko" in x or "trem2 ko" in x:
        return GROUP_NAME_TREM2
    elif "wt" in x:
        return GROUP_NAME_WT
    else:
        return "Unknown"

# -------------------------
# Load expression
# -------------------------
if not os.path.exists(EXPR_FP):
    print(f"ERROR: expression file not found: {EXPR_FP}", file=sys.stderr)
    sys.exit(1)

df_expr = pd.read_csv(EXPR_FP, sep="\t", header=0)
# Expect genes in columns, first column possibly gene ids. This script assumes:
# - first column is gene id (or similar) and remaining columns are samples
# - adapt if your file layout differs.
gene_col = df_expr.columns[0]
sample_columns = df_expr.columns[1:].tolist()

# Build expression DataFrame: rows=samples, columns=genes
expr = df_expr.iloc[:, 1:].T.copy()
expr.columns = df_expr.iloc[:, 0].astype(str).values  # gene IDs as columns
expr.index = sample_columns  # sample names as index

# -------------------------
# Load metadata and align
# -------------------------
if not os.path.exists(META_FP):
    print(f"ERROR: metadata file not found: {META_FP}", file=sys.stderr)
    sys.exit(1)

meta = pd.read_csv(META_FP, sep="\t", header=0, dtype=str)
if META_INDEX_COL not in meta.columns:
    raise RuntimeError(f"metadata missing expected index column: {META_INDEX_COL}")

meta = meta.set_index(META_INDEX_COL)

# Align metadata to the expression sample order
# If sample names in expr.index are refinebio accession codes, this will match directly.
# If not, adapt your sample naming / matching logic here.
meta_for_expr = meta.reindex(expr.index)

# derive groups
raw_groups = meta_for_expr[GROUP_COL_IN_META].fillna("Unknown")
groups = raw_groups.map(assign_group)
print("Group counts (including Unknown):\n", groups.value_counts())

# Keep only Trem2KO & WT
valid_idx = groups[groups.isin([GROUP_NAME_TREM2, GROUP_NAME_WT])].index
expr = expr.loc[valid_idx]
groups = groups.loc[valid_idx]

print("After filtering to Trem2KO and WT:")
print(groups.value_counts())

if groups.value_counts().get(GROUP_NAME_TREM2, 0) < MIN_SAMPLES_PER_GROUP or \
   groups.value_counts().get(GROUP_NAME_WT, 0) < MIN_SAMPLES_PER_GROUP:
    raise RuntimeError("Not enough samples in one or both groups for t-test. "
                       "Check metadata group assignment and sample names.")

# -------------------------
# Optional log-transform
# -------------------------
if LOG_TRANSFORM:
    expr_log = np.log2(expr.astype(float) + 1.0)
else:
    expr_log = expr.astype(float)

# -------------------------
# Differential test per gene
# -------------------------
genes = expr_log.columns
results = []
group_mask_trem2 = (groups == GROUP_NAME_TREM2).values
group_mask_wt = (groups == GROUP_NAME_WT).values

expr_array = expr_log.values  # shape (n_samples, n_genes)

for gi, gname in enumerate(genes):
    col = expr_array[:, gi]
    group1_vals = col[group_mask_trem2]
    group2_vals = col[group_mask_wt]
    # if any group all-NaN skip
    if np.isnan(group1_vals).all() or np.isnan(group2_vals).all():
        pval = np.nan
        tstat = np.nan
    else:
        # Welch's t-test
        try:
            tstat, pval = stats.ttest_ind(group1_vals, group2_vals, equal_var=False, nan_policy='omit')
        except Exception:
            tstat, pval = np.nan, np.nan
    mean_trem2 = np.nanmean(group1_vals)
    mean_wt = np.nanmean(group2_vals)
    # log2 fold change using log-transformed data -> difference of means is log2FC
    log2fc = mean_trem2 - mean_wt
    results.append((gname, mean_wt, mean_trem2, log2fc, tstat, pval))

res_df = pd.DataFrame(results, columns=[
    "gene", "mean_WT", "mean_Trem2KO", "log2FC", "t_stat", "pvalue"
])
# multiple testing correction
res_df["padj"] = np.nan
valid_p = ~res_df["pvalue"].isna()
if valid_p.sum() > 0:
    rej, padj, _, _ = multipletests(res_df.loc[valid_p, "pvalue"].values, method="fdr_bh")
    res_df.loc[valid_p, "padj"] = padj
else:
    res_df["padj"] = np.nan

# add significance flag
res_df["significant"] = (res_df["padj"] < PADJ_CUTOFF) & (res_df["log2FC"].abs() >= LFC_CUTOFF)

# sort and save
res_df = res_df.sort_values(by="padj", na_position="last")
outfp = "results/differential_expression_data.tsv"
res_df.to_csv(outfp, sep="\t", index=False)
print(f"Wrote differential results to: {outfp}")

# -------------------------
# Volcano plot (bioinfokit if present, else matplotlib)
# -------------------------
try:
    from bioinfokit import visuz

    # Bioinfokit expects specific column names: 'log2fc' and 'pval'
    vdf = res_df.rename(
        columns={"gene": "genes", "log2FC": "log2fc", "pvalue": "pval"}
    )

    # Volcano plot
    visuz.GeneExpression.volcano(
        df=vdf,
        lfc="log2fc",   # name of log2FC column
        pv="pval",      # name of p-value column
        geneid="genes", # optional: annotate top genes
        lfc_thr=(0.25, 2),
        pv_thr=(1e-200, 0.1),
        color=("#1f77b4", "grey", "#d62728"),
        show=False
    )

    # Save
    plt.gcf().set_size_inches(7, 6)
    plt.title("Volcano: Trem2KO vs WT")
    plt.savefig("plots/volcano_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved volcano plot using bioinfokit -> plots/volcano_plot.png")
except Exception as e:
    from matplotlib.patches import Patch  # add this import at the top of your script

    # fallback: matplotlib volcano with legend
    print("bioinfokit volcano failed or not installed; using matplotlib.")

    # compute -log10(p-value)
    res_df["-log10_pval"] = -np.log10(res_df["pvalue"].replace(0, np.nextafter(0, 1)))

    # determine colors based on LFC and padj thresholds
    def volcano_color(row):
        if row["padj"] <= PADJ_CUTOFF and abs(row["log2FC"]) >= LFC_CUTOFF:
            return "red" if row["log2FC"] > 0 else "blue"
        else:
            return "grey"

    res_df["color"] = res_df.apply(volcano_color, axis=1)

    plt.figure(figsize=(7,6))
    sns.scatterplot(
        x="log2FC", 
        y="-log10_pval", 
        data=res_df,
        hue="color",
        palette={"red": "red", "blue": "blue", "grey": "grey"},
        legend=False,  # we’ll make a custom legend
        s=20
    )

    # label threshold lines
    plt.axhline(-np.log10(PVALUE_CUTOFF), linestyle="--", linewidth=0.8, color="black")
    plt.axvline(LFC_CUTOFF, linestyle="--", linewidth=0.8, color="black")
    plt.axvline(-LFC_CUTOFF, linestyle="--", linewidth=0.8, color="black")

    plt.xlabel("log2 Fold Change (Trem2KO - WT)")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano: Trem2KO vs WT")

    # custom legend
    legend_elements = [
        Patch(facecolor="red", edgecolor="k", label=f"Upregulated (padj<{PADJ_CUTOFF}, LFC>{LFC_CUTOFF})"),
        Patch(facecolor="blue", edgecolor="k", label=f"Downregulated (padj<{PADJ_CUTOFF}, LFC<-{LFC_CUTOFF})"),
        Patch(facecolor="grey", edgecolor="k", label="Not significant")
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/volcano.png", dpi=150)
    plt.close()
    print("Saved volcano plot -> plots/volcano.png")



# -------------------------
# Heatmap of top genes
# -------------------------
top_genes = res_df.dropna(subset=["padj"]).sort_values("padj").head(TOP_N_GENES_HEATMAP)["gene"].tolist()
if len(top_genes) == 0:
    print("No genes passed padj filter (or none available). Skipping heatmap.")
else:
    heat_df = expr_log[top_genes].copy()
    # z-score genes (rows are samples): transpose to genes x samples for zscore by gene
    heat = (heat_df - heat_df.mean(axis=0)) / heat_df.std(axis=0)
    # create a sample annotation for the group
    sample_df = pd.DataFrame({"Group": groups.loc[heat.index]})
    lut = {GROUP_NAME_TREM2: "#d62728", GROUP_NAME_WT: "#1f77b4"}
    col_colors = sample_df["Group"].map(lut)
    cg = sns.clustermap(heat, row_cluster=True, col_cluster=True,
                        row_colors=col_colors, figsize=(10, max(6, len(top_genes)*0.12)),
                        xticklabels=True, yticklabels=False, cmap="vlag")
    plt.suptitle(f"Top {len(top_genes)} genes by padj", y=1.02)
    plt.savefig("plots/heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap -> plots/heatmap.png")

print("Done.")
