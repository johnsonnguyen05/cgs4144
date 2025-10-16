#!/usr/bin/env python3
"""Generate comprehensive clustering analysis report."""
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def load_clustering_results(results_dir):
    """Load clustering results from directory."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_labels.tsv'):
            method_name = filename.replace('_labels.tsv', '')
            filepath = os.path.join(results_dir, filename)
            
            try:
                df = pd.read_csv(filepath, sep="\t")
                labels = df['cluster'].values
                unique, counts = np.unique(labels, return_counts=True)
                results[method_name] = {
                    'labels': labels,
                    'n_clusters': len(unique),
                    'cluster_counts': dict(zip(unique, counts))
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results

def analyze_cluster_stability(results):
    """Analyze cluster stability across different parameters."""
    print("\n" + "="*60)
    print("CLUSTER STABILITY ANALYSIS")
    print("="*60)
    
    # Group results by method
    method_groups = {}
    for name, result in results.items():
        method = name.split('_')[0]  # Extract method name
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append((name, result))
    
    for method, method_results in method_groups.items():
        print(f"\n{method.upper()} Results:")
        for name, result in method_results:
            print(f"  {name}: {result['n_clusters']} clusters, sizes: {result['cluster_counts']}")

def main():
    # Set up paths
    current_results_dir = "Assignment_3/results"
    varied_k_dir = "Assignment_3/varied_k_results"
    varied_genes_dir = "Assignment_3/varied_genes_results"
    analysis_dir = "Assignment_3/analysis_results"
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE CLUSTERING ANALYSIS REPORT")
    print("="*60)
    
    # Load all results
    all_results = {}
    
    # Current results
    if os.path.exists(current_results_dir):
        print(f"\n1. LOADING CURRENT RESULTS")
        current_results = load_clustering_results(current_results_dir)
        all_results.update(current_results)
        print(f"Loaded {len(current_results)} current results")
    
    # Varied k results
    if os.path.exists(varied_k_dir):
        print(f"\n2. LOADING VARIED K RESULTS")
        varied_k_results = load_clustering_results(varied_k_dir)
        all_results.update(varied_k_results)
        print(f"Loaded {len(varied_k_results)} varied k results")
    
    # Varied genes results
    if os.path.exists(varied_genes_dir):
        print(f"\n3. LOADING VARIED GENES RESULTS")
        varied_genes_results = load_clustering_results(varied_genes_dir)
        all_results.update(varied_genes_results)
        print(f"Loaded {len(varied_genes_results)} varied genes results")
    
    print(f"\nTotal results loaded: {len(all_results)}")
    
    # Analyze cluster stability
    analyze_cluster_stability(all_results)
    
    # Load chi-squared results if available
    chi_squared_file = f"{analysis_dir}/chi_squared_results.tsv"
    if os.path.exists(chi_squared_file):
        print(f"\n4. LOADING CHI-SQUARED RESULTS")
        chi_squared_df = pd.read_csv(chi_squared_file, sep="\t")
        print(f"Loaded {len(chi_squared_df)} chi-squared comparisons")
        
        # Summary statistics
        print(f"\nChi-squared test summary:")
        print(f"  Total comparisons: {len(chi_squared_df)}")
        print(f"  Significant differences (p < 0.05): {sum(chi_squared_df['significant'])}")
        print(f"  Non-significant differences: {len(chi_squared_df) - sum(chi_squared_df['significant'])}")
        
        # Most similar and different
        min_chi2_idx = chi_squared_df['chi2'].idxmin()
        max_chi2_idx = chi_squared_df['chi2'].idxmax()
        
        most_similar = chi_squared_df.iloc[min_chi2_idx]
        most_different = chi_squared_df.iloc[max_chi2_idx]
        
        print(f"\nMost similar clustering:")
        print(f"  {most_similar['method1']} vs {most_similar['method2']}")
        print(f"  chi2 = {most_similar['chi2']:.4f}, p = {most_similar['p_value']:.4f}")
        
        print(f"\nMost different clustering:")
        print(f"  {most_different['method1']} vs {most_different['method2']}")
        print(f"  chi2 = {most_different['chi2']:.4f}, p = {most_different['p_value']:.4f}")
    
    # Generate summary report
    print(f"\n5. GENERATING SUMMARY REPORT")
    
    # Create summary table
    summary_data = []
    for name, result in all_results.items():
        summary_data.append({
            'method': name,
            'n_clusters': result['n_clusters'],
            'cluster_sizes': str(result['cluster_counts'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{analysis_dir}/clustering_summary.tsv", sep="\t", index=False)
    print(f"Saved clustering summary to {analysis_dir}/clustering_summary.tsv")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"\nResults analyzed:")
    print(f"- Current clustering methods: {[name for name in all_results.keys() if not any(x in name for x in ['k', 'genes']) and name in ['gmm', 'kmeans', 'spectral']]}")
    print(f"- Varied k experiments: {[name for name in all_results.keys() if 'k' in name]}")
    print(f"- Varied genes experiments: {[name for name in all_results.keys() if 'genes' in name]}")
    
    print(f"\nKey findings:")
    current_methods = ['gmm', 'kmeans', 'spectral']
    for method in current_methods:
        if method in all_results:
            result = all_results[method]
            print(f"- {method.upper()}: {result['n_clusters']} clusters found")
    
    print(f"\nAll results saved to: {analysis_dir}/")
    print("- clustering_summary.tsv: Summary of all clustering results")
    if os.path.exists(chi_squared_file):
        print("- chi_squared_results.tsv: Detailed chi-squared test results")
        print("- chi2_matrix.tsv: Chi-squared statistic matrix")
        print("- pvalue_matrix.tsv: P-value matrix")

if __name__ == "__main__":
    main()
