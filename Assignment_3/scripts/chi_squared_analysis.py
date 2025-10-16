#!/usr/bin/env python3
""" chi-squared analysis comparing methods within same gene counts."""
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations

def chi_squared_test(labels1, labels2, name1, name2):
    """Perform chi-squared test between two clustering results."""
    contingency_table = pd.crosstab(labels1, labels2)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_value, contingency_table

def load_clustering_results(base_dir):
    """Load clustering results from directory (handles subdirectories)."""
    results = {}
    
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('_labels.tsv'):
                filepath = os.path.join(root, filename)
                # Create method name from path
                relative_path = os.path.relpath(filepath, base_dir)
                method_name = relative_path.replace(os.sep, '_').replace('_labels.tsv', '')
                
                try:
                    df = pd.read_csv(filepath, sep="\t")
                    labels = df['cluster'].values
                    results[method_name] = labels
                    print(f"Loaded {method_name}: {len(labels)} samples, {len(np.unique(labels))} clusters")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return results

def group_by_gene_count(results):
    """Group results by gene count for meaningful comparisons."""
    groups = {}
    
    for name, labels in results.items():
        # Extract gene count from name
        if 'genes' in name:
            # Extract number after 'genes'
            parts = name.split('genes')
            if len(parts) > 1:
                gene_count = parts[1]
                if gene_count not in groups:
                    groups[gene_count] = {}
                groups[gene_count][name] = labels
        elif any(x in name for x in ['gmm', 'kmeans', 'spectral']) and 'genes' not in name:
            # Current results (5000 genes)
            if '5000' not in groups:
                groups['5000'] = {}
            groups['5000'][name] = labels
    
    return groups

def main():
    # Set up paths
    current_results_dir = "Assignment_3/results"
    varied_genes_dir = "Assignment_3/varied_genes_results"
    analysis_dir = "Assignment_3/analysis_results"
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("="*60)
    print("CHI-SQUARED ANALYSIS (SAME GENE COUNTS)")
    print("="*60)
    
    # Load all results
    all_results = {}
    
    if os.path.exists(current_results_dir):
        print(f"\nLoading current results from {current_results_dir}")
        all_results.update(load_clustering_results(current_results_dir))
    
    if os.path.exists(varied_genes_dir):
        print(f"\nLoading varied genes results from {varied_genes_dir}")
        all_results.update(load_clustering_results(varied_genes_dir))
    
    print(f"\nTotal results loaded: {len(all_results)}")
    
    # Group by gene count
    gene_groups = group_by_gene_count(all_results)
    
    print(f"\nGene count groups found: {list(gene_groups.keys())}")
    
    # Perform chi-squared tests
    print("\n" + "="*60)
    print("CHI-SQUARED TESTS")
    print("="*60)
    
    all_chi_squared_results = []
    
    for gene_count, methods in gene_groups.items():
        print(f"\n--- Gene Count: {gene_count} ---")
        print(f"Methods: {list(methods.keys())}")
        
        if len(methods) >= 2:
            # Compare methods within this gene count
            method_names = list(methods.keys())
            for name1, name2 in combinations(method_names, 2):
                labels1 = methods[name1]
                labels2 = methods[name2]
                
                chi2, p_value, contingency_table = chi_squared_test(labels1, labels2, name1, name2)
                
                result = {
                    'gene_count': gene_count,
                    'method1': name1,
                    'method2': name2,
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'contingency_shape': contingency_table.shape
                }
                
                all_chi_squared_results.append(result)
                
                print(f"  {name1} vs {name2}: chi2 = {chi2:.4f}, p = {p_value:.4f}, significant = {p_value < 0.05}")
        else:
            print(f"  Only {len(methods)} method(s) available - no comparisons possible")
    
    # Save results
    if all_chi_squared_results:
        chi_squared_df = pd.DataFrame(all_chi_squared_results)
        chi_squared_path = os.path.join(analysis_dir, "chi_squared_results_table.tsv")
        chi_squared_df.to_csv(chi_squared_path, sep="\t", index=False)
        print(f"\nSaved chi-squared results table to {chi_squared_path}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("CHI-SQUARED ANALYSIS SUMMARY")
        print("="*60)
        
        total_comparisons = len(all_chi_squared_results)
        significant = sum(r['significant'] for r in all_chi_squared_results)
        non_significant = total_comparisons - significant
        
        print(f"Total comparisons: {total_comparisons}")
        print(f"Significant differences (p < 0.05): {significant}")
        print(f"Non-significant differences: {non_significant}")
        
        # Summary by gene count
        print(f"\nSummary by gene count:")
        for gene_count in gene_groups.keys():
            gene_results = [r for r in all_chi_squared_results if r['gene_count'] == gene_count]
            if gene_results:
                gene_significant = sum(r['significant'] for r in gene_results)
                print(f"  {gene_count} genes: {len(gene_results)} comparisons, {gene_significant} significant")
        
        # Most similar and different within each gene count
        for gene_count in gene_groups.keys():
            gene_results = [r for r in all_chi_squared_results if r['gene_count'] == gene_count]
            if gene_results:
                min_chi2 = min(r['chi2'] for r in gene_results)
                max_chi2 = max(r['chi2'] for r in gene_results)
                
                most_similar = next(r for r in gene_results if r['chi2'] == min_chi2)
                most_different = next(r for r in gene_results if r['chi2'] == max_chi2)
                
                print(f"\n{gene_count} genes - Most similar:")
                print(f"  {most_similar['method1']} vs {most_similar['method2']}")
                print(f"  chi2 = {min_chi2:.4f}, p = {most_similar['p_value']:.4f}")
                
                print(f"\n{gene_count} genes - Most different:")
                print(f"  {most_different['method1']} vs {most_different['method2']}")
                print(f"  chi2 = {max_chi2:.4f}, p = {most_different['p_value']:.4f}")
    else:
        print("No valid comparisons found!")

if __name__ == "__main__":
    main()
