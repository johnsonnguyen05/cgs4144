#!/usr/bin/env python3
"""
Gene Set Enrichment Analysis (GSEA) using PyDESeq2 results
This script performs GSEA on differentially expressed genes using GO ontology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gseapy as gp
from gseapy.plot import gseaplot, dotplot
import warnings
warnings.filterwarnings('ignore')

def load_differential_expression_results(file_path):
    """Load differential expression results from PyDESeq2"""
    print("Loading differential expression results...")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} genes")
    print(f"Columns: {list(df.columns)}")
    return df

def prepare_gene_ranking(df, ranking_metric='log2FoldChange'):
    """Prepare gene ranking for GSEA analysis"""
    print(f"Preparing gene ranking using {ranking_metric}...")
    
    # Filter out genes with missing values
    df_clean = df.dropna(subset=[ranking_metric, 'padj'])
    
    # Create gene ranking based on log2FoldChange
    # Sort by log2FoldChange in descending order (most upregulated first)
    df_ranked = df_clean.sort_values(ranking_metric, ascending=False)
    
    # Create gene ranking dictionary
    gene_ranking = dict(zip(df_ranked['Gene'], df_ranked[ranking_metric]))
    
    print(f"Gene ranking created for {len(gene_ranking)} genes")
    print(f"Top 5 upregulated genes: {list(gene_ranking.keys())[:5]}")
    print(f"Top 5 downregulated genes: {list(gene_ranking.keys())[-5:]}")
    
    return gene_ranking, df_ranked

def run_gsea_analysis(gene_ranking, organism='mouse'):
    """Run Gene Set Enrichment Analysis using GO ontology"""
    print("Running GSEA analysis...")
    
    # Define gene sets to test (GO terms)
    gene_sets = [
        'GO_Biological_Process_2023',
        'GO_Cellular_Component_2023', 
        'GO_Molecular_Function_2023',
        'KEGG_2021_Mouse',
        'Reactome_2022'
    ]
    
    # Run GSEA using prerank method (no class labels needed)
    gsea_results = {}
    
    for gene_set in gene_sets:
        print(f"Analyzing {gene_set}...")
        try:
            # Run GSEA using prerank method
            gsea = gp.prerank(
                rnk=gene_ranking,
                gene_sets=gene_set,
                organism=organism,
                outdir=None,
                no_plot=True,
                min_size=5,
                max_size=500,
                permutation_num=1000,
                weighted_score_type=1,
                verbose=True
            )
            gsea_results[gene_set] = gsea
            print(f"✓ {gene_set}: {len(gsea.res2d)} significant gene sets")
        except Exception as e:
            print(f"✗ Error with {gene_set}: {str(e)}")
            continue
    
    return gsea_results

def analyze_significant_genes(df, padj_threshold=0.05, lfc_threshold=1.0):
    """Analyze significantly differentially expressed genes"""
    print(f"Analyzing significant genes (padj < {padj_threshold}, |log2FC| > {lfc_threshold})...")
    
    # Filter significant genes
    significant = df[
        (df['padj'] < padj_threshold) & 
        (np.abs(df['log2FoldChange']) > lfc_threshold)
    ].copy()
    
    # Separate up and down regulated
    upregulated = significant[significant['log2FoldChange'] > 0]
    downregulated = significant[significant['log2FoldChange'] < 0]
    
    print(f"Total significant genes: {len(significant)}")
    print(f"Upregulated: {len(upregulated)}")
    print(f"Downregulated: {len(downregulated)}")
    
    return significant, upregulated, downregulated

def run_over_representation_analysis(upregulated_genes, downregulated_genes, organism='mouse'):
    """Run Over-Representation Analysis (ORA) on significant genes"""
    print("Running Over-Representation Analysis...")
    
    ora_results = {}
    
    # Analyze upregulated genes
    if len(upregulated_genes) > 0:
        print(f"Analyzing {len(upregulated_genes)} upregulated genes...")
        try:
            ora_up = gp.enrichr(
                gene_list=upregulated_genes['Gene'].tolist(),
                gene_sets=['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023'],
                organism=organism,
                outdir=None,
                no_plot=True
            )
            ora_results['upregulated'] = ora_up
            print(f"✓ Upregulated ORA: {len(ora_up.res2d)} enriched terms")
        except Exception as e:
            print(f"✗ Error with upregulated ORA: {str(e)}")
    
    # Analyze downregulated genes
    if len(downregulated_genes) > 0:
        print(f"Analyzing {len(downregulated_genes)} downregulated genes...")
        try:
            ora_down = gp.enrichr(
                gene_list=downregulated_genes['Gene'].tolist(),
                gene_sets=['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023'],
                organism=organism,
                outdir=None,
                no_plot=True
            )
            ora_results['downregulated'] = ora_down
            print(f"✓ Downregulated ORA: {len(ora_down.res2d)} enriched terms")
        except Exception as e:
            print(f"✗ Error with downregulated ORA: {str(e)}")
    
    return ora_results

def create_visualizations(gsea_results, ora_results, output_dir='results'):
    """Create visualizations for GSEA and ORA results"""
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. GSEA Results Summary
    if gsea_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gene Set Enrichment Analysis Results', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for gene_set, gsea in gsea_results.items():
            if plot_idx < 4 and len(gsea.res2d) > 0:
                ax = axes[plot_idx // 2, plot_idx % 2]
                
                # Plot top 10 most significant terms
                top_terms = gsea.res2d.head(10)
                if len(top_terms) > 0:
                    y_pos = np.arange(len(top_terms))
                    bars = ax.barh(y_pos, -np.log10(top_terms['Adjusted P-value']))
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([term[:50] + '...' if len(term) > 50 else term 
                                      for term in top_terms['Term']], fontsize=8)
                    ax.set_xlabel('-log10(Adjusted P-value)')
                    ax.set_title(f'{gene_set}\nTop 10 Enriched Terms')
                    ax.grid(True, alpha=0.3)
                    
                    # Color bars by significance
                    for i, bar in enumerate(bars):
                        pval = top_terms.iloc[i]['Adjusted P-value']
                        if pval < 0.001:
                            bar.set_color('red')
                        elif pval < 0.01:
                            bar.set_color('orange')
                        else:
                            bar.set_color('lightblue')
                
                plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, 4):
            axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gsea_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. ORA Results Summary
    if ora_results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Over-Representation Analysis Results', fontsize=16, fontweight='bold')
        
        for i, (direction, ora) in enumerate(ora_results.items()):
            if len(ora.res2d) > 0:
                ax = axes[i]
                top_terms = ora.res2d.head(15)
                
                y_pos = np.arange(len(top_terms))
                bars = ax.barh(y_pos, -np.log10(top_terms['Adjusted P-value']))
                ax.set_yticks(y_pos)
                ax.set_yticklabels([term[:40] + '...' if len(term) > 40 else term 
                                  for term in top_terms['Term']], fontsize=9)
                ax.set_xlabel('-log10(Adjusted P-value)')
                ax.set_title(f'{direction.capitalize()} Genes\nTop 15 Enriched Terms')
                ax.grid(True, alpha=0.3)
                
                # Color bars by significance
                for j, bar in enumerate(bars):
                    pval = top_terms.iloc[j]['Adjusted P-value']
                    if pval < 0.001:
                        bar.set_color('red')
                    elif pval < 0.01:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightblue')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ora_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

def save_results(gsea_results, ora_results, significant_genes, output_dir='results'):
    """Save all results to files"""
    print("Saving results...")
    
    # Save GSEA results
    if gsea_results:
        for gene_set, gsea in gsea_results.items():
            if len(gsea.res2d) > 0:
                filename = f"{output_dir}/gsea_{gene_set.replace(' ', '_').replace('/', '_')}.tsv"
                gsea.res2d.to_csv(filename, sep='\t', index=False)
                print(f"Saved GSEA results: {filename}")
    
    # Save ORA results
    if ora_results:
        for direction, ora in ora_results.items():
            if len(ora.res2d) > 0:
                filename = f"{output_dir}/ora_{direction}_genes.tsv"
                ora.res2d.to_csv(filename, sep='\t', index=False)
                print(f"Saved ORA results: {filename}")
    
    # Save significant genes
    if len(significant_genes) > 0:
        filename = f"{output_dir}/significant_genes_for_gsea.tsv"
        significant_genes.to_csv(filename, sep='\t', index=False)
        print(f"Saved significant genes: {filename}")

def main():
    """Main function to run the complete GSEA analysis"""
    print("=" * 60)
    print("GENE SET ENRICHMENT ANALYSIS (GSEA)")
    print("=" * 60)
    
    # Load data
    df = load_differential_expression_results('results/SRP119064_diff_expr_results.tsv')
    
    # Prepare gene ranking
    gene_ranking, df_ranked = prepare_gene_ranking(df)
    
    # Analyze significant genes
    significant_genes, upregulated, downregulated = analyze_significant_genes(df)
    
    # Run GSEA analysis
    gsea_results = run_gsea_analysis(gene_ranking)
    
    # Run ORA analysis
    ora_results = run_over_representation_analysis(upregulated, downregulated)
    
    # Create visualizations
    create_visualizations(gsea_results, ora_results)
    
    # Save results
    save_results(gsea_results, ora_results, significant_genes)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Results saved in 'results/' directory")
    print("Check the generated plots and TSV files for detailed results")

if __name__ == "__main__":
    main()
