#!/usr/bin/env python3
"""
Disease Ontology Enrichment Analysis using PyDESeq2 results
This script performs enrichment analysis on differentially expressed genes using Disease Ontology
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
    """Prepare gene ranking for enrichment analysis"""
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

def run_disease_enrichment_analysis(gene_ranking, organism='mouse'):
    """Run Disease Ontology enrichment analysis"""
    print("Running Disease Ontology enrichment analysis...")
    
    # Define disease-related gene sets
    disease_gene_sets = [
        'DisGeNET',
        'Disease_Perturbations_from_GEO_down',
        'Disease_Perturbations_from_GEO_up',
        'Disease_Signatures_from_GEO_down_2014',
        'Disease_Signatures_from_GEO_up_2014',
        'OMIM_Disease',
        'OMIM_Expanded',
        'PhenGenI_Association_2014',
        'PhenGenI_Association_2014_genes',
        'PhenGenI_Association_2014_variants',
        'ClinVar_2019',
        'ClinVar_2019_Cancer',
        'ClinVar_2019_Cardiovascular',
        'ClinVar_2019_Immune',
        'ClinVar_2019_Metabolic',
        'ClinVar_2019_Neurological',
        'ClinVar_2019_Psychiatric'
    ]
    
    # Run enrichment analysis using prerank method
    enrichment_results = {}
    
    for gene_set in disease_gene_sets:
        print(f"Analyzing {gene_set}...")
        try:
            # Run enrichment analysis using prerank method
            enrichment = gp.prerank(
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
            enrichment_results[gene_set] = enrichment
            print(f"✓ {gene_set}: {len(enrichment.res2d)} significant disease associations")
        except Exception as e:
            print(f"✗ Error with {gene_set}: {str(e)}")
            continue
    
    return enrichment_results

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

def run_disease_over_representation_analysis(upregulated_genes, downregulated_genes, organism='mouse'):
    """Run Disease Over-Representation Analysis (ORA) on significant genes"""
    print("Running Disease Over-Representation Analysis...")
    
    # Disease-specific gene sets for ORA
    disease_gene_sets = [
        'DisGeNET',
        'OMIM_Disease',
        'ClinVar_2019',
        'Disease_Perturbations_from_GEO_up',
        'Disease_Perturbations_from_GEO_down'
    ]
    
    ora_results = {}
    
    # Analyze upregulated genes
    if len(upregulated_genes) > 0:
        print(f"Analyzing {len(upregulated_genes)} upregulated genes for disease associations...")
        try:
            ora_up = gp.enrichr(
                gene_list=upregulated_genes['Gene'].tolist(),
                gene_sets=disease_gene_sets,
                organism=organism,
                outdir=None,
                no_plot=True
            )
            ora_results['upregulated'] = ora_up
            print(f"✓ Upregulated disease ORA: {len(ora_up.res2d)} enriched disease terms")
        except Exception as e:
            print(f"✗ Error with upregulated disease ORA: {str(e)}")
    
    # Analyze downregulated genes
    if len(downregulated_genes) > 0:
        print(f"Analyzing {len(downregulated_genes)} downregulated genes for disease associations...")
        try:
            ora_down = gp.enrichr(
                gene_list=downregulated_genes['Gene'].tolist(),
                gene_sets=disease_gene_sets,
                organism=organism,
                outdir=None,
                no_plot=True
            )
            ora_results['downregulated'] = ora_down
            print(f"✓ Downregulated disease ORA: {len(ora_down.res2d)} enriched disease terms")
        except Exception as e:
            print(f"✗ Error with downregulated disease ORA: {str(e)}")
    
    return ora_results

def create_disease_visualizations(enrichment_results, ora_results, output_dir='results'):
    """Create visualizations for disease enrichment analysis results"""
    print("Creating disease enrichment visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. Disease Enrichment Results Summary
    if enrichment_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Disease Ontology Enrichment Analysis Results', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for gene_set, enrichment in enrichment_results.items():
            if plot_idx < 4 and len(enrichment.res2d) > 0:
                ax = axes[plot_idx // 2, plot_idx % 2]
                
                # Plot top 10 most significant disease associations
                top_terms = enrichment.res2d.head(10)
                if len(top_terms) > 0:
                    y_pos = np.arange(len(top_terms))
                    bars = ax.barh(y_pos, -np.log10(top_terms['Adjusted P-value']))
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([term[:50] + '...' if len(term) > 50 else term 
                                      for term in top_terms['Term']], fontsize=8)
                    ax.set_xlabel('-log10(Adjusted P-value)')
                    ax.set_title(f'{gene_set}\nTop 10 Disease Associations')
                    ax.grid(True, alpha=0.3)
                    
                    # Color bars by significance
                    for i, bar in enumerate(bars):
                        pval = top_terms.iloc[i]['Adjusted P-value']
                        if pval < 0.001:
                            bar.set_color('darkred')
                        elif pval < 0.01:
                            bar.set_color('red')
                        else:
                            bar.set_color('lightcoral')
                
                plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, 4):
            axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/disease_enrichment_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Disease ORA Results Summary
    if ora_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Disease Over-Representation Analysis Results', fontsize=16, fontweight='bold')
        
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
                ax.set_title(f'{direction.capitalize()} Genes\nTop 15 Disease Associations')
                ax.grid(True, alpha=0.3)
                
                # Color bars by significance
                for j, bar in enumerate(bars):
                    pval = top_terms.iloc[j]['Adjusted P-value']
                    if pval < 0.001:
                        bar.set_color('darkred')
                    elif pval < 0.01:
                        bar.set_color('red')
                    else:
                        bar.set_color('lightcoral')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/disease_ora_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_disease_summary_table(enrichment_results, ora_results, output_dir='results'):
    """Create a comprehensive summary table of disease associations"""
    print("Creating disease association summary table...")
    
    all_results = []
    
    # Collect GSEA results
    for gene_set, enrichment in enrichment_results.items():
        if len(enrichment.res2d) > 0:
            df_temp = enrichment.res2d.copy()
            df_temp['Analysis_Type'] = 'GSEA'
            df_temp['Gene_Set'] = gene_set
            all_results.append(df_temp)
    
    # Collect ORA results
    for direction, ora in ora_results.items():
        if len(ora.res2d) > 0:
            df_temp = ora.res2d.copy()
            df_temp['Analysis_Type'] = 'ORA'
            df_temp['Gene_Set'] = f'{direction}_genes'
            all_results.append(df_temp)
    
    if all_results:
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by significance
        combined_df = combined_df.sort_values('Adjusted P-value')
        
        # Save comprehensive results
        filename = f"{output_dir}/disease_enrichment_comprehensive_results.tsv"
        combined_df.to_csv(filename, sep='\t', index=False)
        print(f"Saved comprehensive disease results: {filename}")
        
        # Create top disease associations summary
        top_diseases = combined_df.head(20)
        filename = f"{output_dir}/top_disease_associations.tsv"
        top_diseases.to_csv(filename, sep='\t', index=False)
        print(f"Saved top disease associations: {filename}")
        
        return combined_df
    
    return None

def save_disease_results(enrichment_results, ora_results, significant_genes, output_dir='results'):
    """Save all disease enrichment results to files"""
    print("Saving disease enrichment results...")
    
    # Save GSEA results
    if enrichment_results:
        for gene_set, enrichment in enrichment_results.items():
            if len(enrichment.res2d) > 0:
                filename = f"{output_dir}/disease_gsea_{gene_set.replace(' ', '_').replace('/', '_')}.tsv"
                enrichment.res2d.to_csv(filename, sep='\t', index=False)
                print(f"Saved disease GSEA results: {filename}")
    
    # Save ORA results
    if ora_results:
        for direction, ora in ora_results.items():
            if len(ora.res2d) > 0:
                filename = f"{output_dir}/disease_ora_{direction}_genes.tsv"
                ora.res2d.to_csv(filename, sep='\t', index=False)
                print(f"Saved disease ORA results: {filename}")
    
    # Save significant genes
    if len(significant_genes) > 0:
        filename = f"{output_dir}/significant_genes_for_disease_analysis.tsv"
        significant_genes.to_csv(filename, sep='\t', index=False)
        print(f"Saved significant genes: {filename}")

def main():
    """Main function to run the complete disease ontology enrichment analysis"""
    print("=" * 70)
    print("DISEASE ONTOLOGY ENRICHMENT ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_differential_expression_results('results/SRP119064_diff_expr_results.tsv')
    
    # Prepare gene ranking
    gene_ranking, df_ranked = prepare_gene_ranking(df)
    
    # Analyze significant genes
    significant_genes, upregulated, downregulated = analyze_significant_genes(df)
    
    # Run disease enrichment analysis
    enrichment_results = run_disease_enrichment_analysis(gene_ranking)
    
    # Run disease ORA analysis
    ora_results = run_disease_over_representation_analysis(upregulated, downregulated)
    
    # Create visualizations
    create_disease_visualizations(enrichment_results, ora_results)
    
    # Create summary table
    create_disease_summary_table(enrichment_results, ora_results)
    
    # Save results
    save_disease_results(enrichment_results, ora_results, significant_genes)
    
    print("\n" + "=" * 70)
    print("DISEASE ONTOLOGY ANALYSIS COMPLETE!")
    print("=" * 70)
    print("Results saved in 'results/' directory")
    print("Check the generated plots and TSV files for disease associations")
    print("Key files:")
    print("- disease_enrichment_comprehensive_results.tsv")
    print("- top_disease_associations.tsv")
    print("- disease_enrichment_summary.png")
    print("- disease_ora_summary.png")

if __name__ == "__main__":
    main()

