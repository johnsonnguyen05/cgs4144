#!/usr/bin/env python3
"""
Compare Gene Set Enrichment and Disease Set Enrichment Results
This script identifies shared significantly enriched terms between GO and Disease analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_enrichment_results(results_dir='results'):
    """Load enrichment results from both analyses"""
    print("Loading enrichment analysis results...")
    
    results = {}
    
    # Try to load GO enrichment results
    go_files = [
        'wilcoxon_go_enrichment_results.tsv',
        'gsea_GO_Biological_Process_2023.tsv',
        'gsea_GO_Cellular_Component_2023.tsv',
        'gsea_GO_Molecular_Function_2023.tsv'
    ]
    
    go_results = []
    for file in go_files:
        file_path = Path(results_dir) / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, sep='\t')
                if 'Term' in df.columns:
                    df['Analysis_Type'] = 'GO'
                    df['Source_File'] = file
                    go_results.append(df)
                    print(f"Loaded GO results: {file} ({len(df)} terms)")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    # Try to load Disease enrichment results
    disease_files = [
        'disease_enrichment_comprehensive_results.tsv',
        'disease_gsea_DisGeNET.tsv',
        'disease_gsea_OMIM_Disease.tsv',
        'disease_gsea_ClinVar_2019.tsv'
    ]
    
    disease_results = []
    for file in disease_files:
        file_path = Path(results_dir) / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, sep='\t')
                if 'Term' in df.columns:
                    df['Analysis_Type'] = 'Disease'
                    df['Source_File'] = file
                    disease_results.append(df)
                    print(f"Loaded Disease results: {file} ({len(df)} terms)")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    results['GO'] = go_results
    results['Disease'] = disease_results
    
    return results

def standardize_columns(df):
    """Standardize column names across different result files"""
    # Common column mappings
    column_mappings = {
        'Adjusted P-value': 'adj_pvalue',
        'Adjusted_P_value': 'adj_pvalue',
        'P-value': 'pvalue',
        'P_value': 'pvalue',
        'Term': 'term',
        'Name': 'term',
        'GO_Name': 'term',
        'Genes_in_GO': 'genes_in_term',
        'Genes_in_term': 'genes_in_term',
        'Size': 'genes_in_term',
        'DEG_in_GO': 'deg_in_term',
        'DEG_in_term': 'deg_in_term',
        'Hits': 'deg_in_term',
        'Enrichment_Ratio': 'enrichment_ratio',
        'Ratio': 'enrichment_ratio',
        'Fold_Enrichment': 'enrichment_ratio',
        'Effect_Size': 'effect_size',
        'NES': 'effect_size',
        'Category': 'category',
        'GO_Category': 'category'
    }
    
    # Rename columns
    df = df.rename(columns=column_mappings)
    
    # Ensure required columns exist
    required_columns = ['term', 'adj_pvalue', 'genes_in_term', 'deg_in_term']
    for col in required_columns:
        if col not in df.columns:
            if col == 'adj_pvalue' and 'pvalue' in df.columns:
                df[col] = df['pvalue']
            elif col == 'genes_in_term' and 'Size' in df.columns:
                df[col] = df['Size']
            elif col == 'deg_in_term' and 'Hits' in df.columns:
                df[col] = df['Hits']
            else:
                df[col] = 0  # Default value
    
    return df

def find_shared_terms(go_results, disease_results, pvalue_threshold=0.05):
    """Find shared significantly enriched terms between GO and Disease analyses"""
    print("Identifying shared significantly enriched terms...")
    
    # Combine all GO results
    all_go_df = pd.concat(go_results, ignore_index=True) if go_results else pd.DataFrame()
    all_disease_df = pd.concat(disease_results, ignore_index=True) if disease_results else pd.DataFrame()
    
    if len(all_go_df) == 0 and len(all_disease_df) == 0:
        print("No enrichment results found!")
        return pd.DataFrame()
    
    # Standardize columns
    if len(all_go_df) > 0:
        all_go_df = standardize_columns(all_go_df)
        all_go_df = all_go_df[all_go_df['adj_pvalue'] < pvalue_threshold]
    
    if len(all_disease_df) > 0:
        all_disease_df = standardize_columns(all_disease_df)
        all_disease_df = all_disease_df[all_disease_df['adj_pvalue'] < pvalue_threshold]
    
    # Find shared terms by name similarity
    shared_terms = []
    
    if len(all_go_df) > 0 and len(all_disease_df) > 0:
        for _, go_row in all_go_df.iterrows():
            go_term = str(go_row['term']).lower()
            
            for _, disease_row in all_disease_df.iterrows():
                disease_term = str(disease_row['term']).lower()
                
                # Check for exact matches or significant overlap
                if (go_term == disease_term or 
                    go_term in disease_term or 
                    disease_term in go_term or
                    self._calculate_similarity(go_term, disease_term) > 0.7):
                    
                    shared_terms.append({
                        'Term_Name': go_row['term'],
                        'GO_Pvalue': go_row['adj_pvalue'],
                        'GO_Genes_in_Term': go_row.get('genes_in_term', 0),
                        'GO_DEG_in_Term': go_row.get('deg_in_term', 0),
                        'GO_Enrichment_Ratio': go_row.get('enrichment_ratio', 0),
                        'GO_Effect_Size': go_row.get('effect_size', 0),
                        'Disease_Pvalue': disease_row['adj_pvalue'],
                        'Disease_Genes_in_Term': disease_row.get('genes_in_term', 0),
                        'Disease_DEG_in_Term': disease_row.get('deg_in_term', 0),
                        'Disease_Enrichment_Ratio': disease_row.get('enrichment_ratio', 0),
                        'Disease_Effect_Size': disease_row.get('effect_size', 0),
                        'GO_Source': go_row.get('Source_File', 'Unknown'),
                        'Disease_Source': disease_row.get('Source_File', 'Unknown'),
                        'Similarity_Score': self._calculate_similarity(go_term, disease_term)
                    })
    
    # Create DataFrame and remove duplicates
    shared_df = pd.DataFrame(shared_terms)
    if len(shared_df) > 0:
        shared_df = shared_df.drop_duplicates(subset=['Term_Name'])
        shared_df = shared_df.sort_values(['GO_Pvalue', 'Disease_Pvalue'])
    
    print(f"Found {len(shared_df)} shared significantly enriched terms")
    return shared_df

def _calculate_similarity(term1, term2):
    """Calculate similarity between two terms using simple word overlap"""
    words1 = set(term1.split())
    words2 = set(term2.split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def create_comparison_visualizations(shared_df, output_dir='results'):
    """Create visualizations comparing GO and Disease enrichment results"""
    print("Creating comparison visualizations...")
    
    if len(shared_df) == 0:
        print("No shared terms to visualize")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. P-value comparison scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Shared Enriched Terms: GO vs Disease Analysis', fontsize=16, fontweight='bold')
    
    # P-value comparison
    ax = axes[0, 0]
    ax.scatter(shared_df['GO_Pvalue'], shared_df['Disease_Pvalue'], 
               c=shared_df['Similarity_Score'], cmap='viridis', alpha=0.7, s=60)
    ax.set_xlabel('GO Analysis P-value')
    ax.set_ylabel('Disease Analysis P-value')
    ax.set_title('P-value Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot([0.001, 0.05], [0.001, 0.05], 'r--', alpha=0.5, label='Equal significance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Term Similarity Score')
    
    # Effect size comparison
    ax = axes[0, 1]
    if 'GO_Effect_Size' in shared_df.columns and 'Disease_Effect_Size' in shared_df.columns:
        ax.scatter(shared_df['GO_Effect_Size'], shared_df['Disease_Effect_Size'], 
                   c=shared_df['Similarity_Score'], cmap='viridis', alpha=0.7, s=60)
        ax.set_xlabel('GO Analysis Effect Size')
        ax.set_ylabel('Disease Analysis Effect Size')
        ax.set_title('Effect Size Comparison')
        ax.plot([-2, 2], [-2, 2], 'r--', alpha=0.5, label='Equal effect')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Enrichment ratio comparison
    ax = axes[1, 0]
    ax.scatter(shared_df['GO_Enrichment_Ratio'], shared_df['Disease_Enrichment_Ratio'], 
               c=shared_df['Similarity_Score'], cmap='viridis', alpha=0.7, s=60)
    ax.set_xlabel('GO Analysis Enrichment Ratio')
    ax.set_ylabel('Disease Analysis Enrichment Ratio')
    ax.set_title('Enrichment Ratio Comparison')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal enrichment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top shared terms
    ax = axes[1, 1]
    top_terms = shared_df.head(10)
    y_pos = np.arange(len(top_terms))
    
    # Create grouped bar chart
    width = 0.35
    ax.barh(y_pos - width/2, -np.log10(top_terms['GO_Pvalue']), width, 
            label='GO Analysis', alpha=0.7)
    ax.barh(y_pos + width/2, -np.log10(top_terms['Disease_Pvalue']), width, 
            label='Disease Analysis', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([term[:30] + '...' if len(term) > 30 else term 
                      for term in top_terms['Term_Name']], fontsize=8)
    ax.set_xlabel('-log10(P-value)')
    ax.set_title('Top 10 Shared Terms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shared_enrichment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(shared_df, output_dir='results'):
    """Create summary statistics for shared terms"""
    print("Creating summary statistics...")
    
    if len(shared_df) == 0:
        print("No shared terms for summary statistics")
        return
    
    summary_stats = {
        'Total_Shared_Terms': len(shared_df),
        'Mean_GO_Pvalue': shared_df['GO_Pvalue'].mean(),
        'Mean_Disease_Pvalue': shared_df['Disease_Pvalue'].mean(),
        'Mean_Similarity_Score': shared_df['Similarity_Score'].mean(),
        'Highly_Similar_Terms': len(shared_df[shared_df['Similarity_Score'] > 0.8]),
        'Significant_in_Both': len(shared_df[(shared_df['GO_Pvalue'] < 0.01) & 
                                            (shared_df['Disease_Pvalue'] < 0.01)]),
        'Mean_GO_Enrichment_Ratio': shared_df['GO_Enrichment_Ratio'].mean(),
        'Mean_Disease_Enrichment_Ratio': shared_df['Disease_Enrichment_Ratio'].mean()
    }
    
    # Save summary statistics
    summary_df = pd.DataFrame([summary_stats])
    filename = f"{output_dir}/shared_terms_summary_statistics.tsv"
    summary_df.to_csv(filename, sep='\t', index=False)
    print(f"Saved summary statistics: {filename}")
    
    return summary_stats

def main():
    """Main function to compare enrichment results"""
    print("=" * 80)
    print("COMPARING GENE SET ENRICHMENT AND DISEASE SET ENRICHMENT RESULTS")
    print("=" * 80)
    
    # Load results
    results = load_enrichment_results()
    
    # Find shared terms
    shared_df = find_shared_terms(results['GO'], results['Disease'])
    
    if len(shared_df) == 0:
        print("No shared significantly enriched terms found between GO and Disease analyses")
        return
    
    # Create visualizations
    create_comparison_visualizations(shared_df)
    
    # Create summary statistics
    summary_stats = create_summary_statistics(shared_df)
    
    # Save shared terms table
    filename = "results/shared_enriched_terms_comparison.csv"
    shared_df.to_csv(filename, index=False)
    print(f"Saved shared terms comparison: {filename}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("SHARED ENRICHMENT ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Total shared terms: {len(shared_df)}")
    print(f"Mean similarity score: {shared_df['Similarity_Score'].mean():.3f}")
    print(f"Terms significant in both analyses: {len(shared_df[(shared_df['GO_Pvalue'] < 0.01) & (shared_df['Disease_Pvalue'] < 0.01)])}")
    print("\nKey files created:")
    print("- shared_enriched_terms_comparison.csv")
    print("- shared_terms_summary_statistics.tsv")
    print("- shared_enrichment_comparison.png")
    
    # Display top shared terms
    if len(shared_df) > 0:
        print("\nTop 5 shared enriched terms:")
        top_5 = shared_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Term_Name']}")
            print(f"   GO P-value: {row['GO_Pvalue']:.2e}, Disease P-value: {row['Disease_Pvalue']:.2e}")
            print(f"   Similarity: {row['Similarity_Score']:.3f}")

if __name__ == "__main__":
    main()

