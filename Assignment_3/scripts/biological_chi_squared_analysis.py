#!/usr/bin/env python3
"""Chi-squared analysis comparing clustering results with biological groups (trem/wild) and between methods."""
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
from statsmodels.stats.multitest import multipletests

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

def load_biological_groups():
    """Load biological groups (trem/wild) from Assignment 1 metadata."""
    metadata_file = "SRP119064/metadata_SRP119064.tsv"
    
    if not os.path.exists(metadata_file):
        print(f"Warning: {metadata_file} not found. Cannot load biological groups.")
        return None
    
    try:
        metadata = pd.read_csv(metadata_file, sep="\t")
        print(f"Loaded metadata: {len(metadata)} samples")
        
        # Extract biological groups from refinebio_subject column
        if 'refinebio_subject' in metadata.columns:
            # Extract trem/wild from subject descriptions
            subject_descriptions = metadata['refinebio_subject'].values
            biological_groups = []
            
            for desc in subject_descriptions:
                if 'trem2ko' in str(desc).lower():
                    biological_groups.append('trem')
                elif 'wt' in str(desc).lower():
                    biological_groups.append('wild')
                else:
                    biological_groups.append('unknown')
            
            biological_groups = np.array(biological_groups)
            print(f"Found biological groups in refinebio_subject column")
        else:
            print("Could not find refinebio_subject column in metadata")
            return None
        
        print(f"Biological groups: {np.unique(biological_groups)}")
        return biological_groups
        
    except Exception as e:
        print(f"Error loading biological groups: {e}")
        return None

def main():
    # Set up paths
    current_results_dir = "Assignment_3/results"
    varied_genes_dir = "Assignment_3/varied_genes_results"
    analysis_dir = "Assignment_3/analysis_results"
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("="*60)
    print("BIOLOGICAL CHI-SQUARED ANALYSIS")
    print("="*60)
    
    # Load clustering results
    all_results = {}
    
    if os.path.exists(current_results_dir):
        print(f"\nLoading current results from {current_results_dir}")
        all_results.update(load_clustering_results(current_results_dir))
    
    if os.path.exists(varied_genes_dir):
        print(f"\nLoading varied genes results from {varied_genes_dir}")
        all_results.update(load_clustering_results(varied_genes_dir))
    
    print(f"\nTotal clustering results loaded: {len(all_results)}")
    
    # Load biological groups
    biological_groups = load_biological_groups()
    
    if biological_groups is None:
        print("Cannot proceed without biological groups. Please check metadata file.")
        return
    
    # Perform chi-squared tests
    print("\n" + "="*60)
    print("CHI-SQUARED TESTS")
    print("="*60)
    
    all_chi_squared_results = []
    
    # 1. Compare each clustering result with biological groups
    print("\n--- Comparing clustering results with biological groups (trem/wild) ---")
    for method_name, cluster_labels in all_results.items():
        # Ensure same length
        min_len = min(len(cluster_labels), len(biological_groups))
        cluster_labels = cluster_labels[:min_len]
        bio_groups = biological_groups[:min_len]
        
        chi2, p_value, contingency_table = chi_squared_test(cluster_labels, bio_groups, method_name, "biological_groups")
        
        result = {
            'comparison_type': 'clustering_vs_biological',
            'method1': method_name,
            'method2': 'biological_groups',
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_shape': contingency_table.shape
        }
        
        all_chi_squared_results.append(result)
        
        print(f"  {method_name} vs biological groups: chi2 = {chi2:.4f}, p = {p_value:.4f}, significant = {p_value < 0.05}")
    
    # 2. Pairwise comparisons between clustering methods
    print("\n--- Pairwise comparisons between clustering methods ---")
    method_names = list(all_results.keys())
    
    for name1, name2 in combinations(method_names, 2):
        labels1 = all_results[name1]
        labels2 = all_results[name2]
        
        # Ensure same length
        min_len = min(len(labels1), len(labels2))
        labels1 = labels1[:min_len]
        labels2 = labels2[:min_len]
        
        chi2, p_value, contingency_table = chi_squared_test(labels1, labels2, name1, name2)
        
        result = {
            'comparison_type': 'method_vs_method',
            'method1': name1,
            'method2': name2,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_shape': contingency_table.shape
        }
        
        all_chi_squared_results.append(result)
        
        print(f"  {name1} vs {name2}: chi2 = {chi2:.4f}, p = {p_value:.4f}, significant = {p_value < 0.05}")
    
    # Save results
    if all_chi_squared_results:
        chi_squared_df = pd.DataFrame(all_chi_squared_results)
        chi_squared_path = os.path.join(analysis_dir, "biological_chi_squared_results.tsv")
        chi_squared_df.to_csv(chi_squared_path, sep="\t", index=False)
        print(f"\nSaved chi-squared results to {chi_squared_path}")
        
        # Apply multiple testing correction
        print("\n" + "="*60)
        print("MULTIPLE TESTING CORRECTION")
        print("="*60)
        
        p_values = chi_squared_df['p_value'].values
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, method='bonferroni')
        
        chi_squared_df['p_corrected'] = p_corrected
        chi_squared_df['significant_corrected'] = rejected
        
        # Save corrected results
        corrected_path = os.path.join(analysis_dir, "biological_chi_squared_corrected.tsv")
        chi_squared_df.to_csv(corrected_path, sep="\t", index=False)
        print(f"Saved corrected results to {corrected_path}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        total_comparisons = len(all_chi_squared_results)
        significant_uncorrected = sum(chi_squared_df['significant'])
        significant_corrected = sum(chi_squared_df['significant_corrected'])
        
        print(f"Total comparisons: {total_comparisons}")
        print(f"Significant differences (uncorrected p < 0.05): {significant_uncorrected}")
        print(f"Significant differences (Bonferroni corrected): {significant_corrected}")
        
        # Summary by comparison type
        clustering_vs_bio = chi_squared_df[chi_squared_df['comparison_type'] == 'clustering_vs_biological']
        method_vs_method = chi_squared_df[chi_squared_df['comparison_type'] == 'method_vs_method']
        
        print(f"\nClustering vs Biological groups:")
        print(f"  Total comparisons: {len(clustering_vs_bio)}")
        print(f"  Significant (uncorrected): {sum(clustering_vs_bio['significant'])}")
        print(f"  Significant (corrected): {sum(clustering_vs_bio['significant_corrected'])}")
        
        print(f"\nMethod vs Method comparisons:")
        print(f"  Total comparisons: {len(method_vs_method)}")
        print(f"  Significant (uncorrected): {sum(method_vs_method['significant'])}")
        print(f"  Significant (corrected): {sum(method_vs_method['significant_corrected'])}")
        
        # Most significant comparisons
        print(f"\nMost significant clustering vs biological:")
        most_sig_bio = clustering_vs_bio.loc[clustering_vs_bio['chi2'].idxmax()]
        print(f"  {most_sig_bio['method1']} vs biological groups")
        print(f"  chi2 = {most_sig_bio['chi2']:.4f}, p = {most_sig_bio['p_value']:.4f}, p_corrected = {most_sig_bio['p_corrected']:.4f}")
        
        print(f"\nMost significant method vs method:")
        most_sig_method = method_vs_method.loc[method_vs_method['chi2'].idxmax()]
        print(f"  {most_sig_method['method1']} vs {most_sig_method['method2']}")
        print(f"  chi2 = {most_sig_method['chi2']:.4f}, p = {most_sig_method['p_value']:.4f}, p_corrected = {most_sig_method['p_corrected']:.4f}")
        
    else:
        print("No valid comparisons found!")

if __name__ == "__main__":
    main()
