#!/usr/bin/env python3
"""Analyze current clustering results and create summary."""
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyze_cluster_counts(labels, method_name):
    """Analyze and print cluster counts."""
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{method_name} cluster counts:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")
    return dict(zip(unique, counts))

def chi_squared_test(labels1, labels2, name1, name2):
    """Perform chi-squared test between two clustering results."""
    # Create contingency table
    contingency_table = pd.crosstab(labels1, labels2)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_value, contingency_table

def main():
    results_dir = "Assignment_3/results"
    
    print("="*60)
    print("ANALYZING CURRENT CLUSTERING RESULTS")
    print("="*60)
    
    # Load current results
    methods = ['gmm', 'kmeans', 'spectral']
    results = {}
    
    for method in methods:
        labels_file = f"{results_dir}/{method}_labels.tsv"
        if os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file, sep="\t")
            labels = labels_df['cluster'].values
            cluster_counts = analyze_cluster_counts(labels, method.upper())
            results[method] = {
                'labels': labels,
                'counts': cluster_counts,
                'n_clusters': len(np.unique(labels))
            }
    
    # Chi-squared tests between current methods
    print("\n" + "="*60)
    print("CHI-SQUARED TESTS BETWEEN CURRENT METHODS")
    print("="*60)
    
    method_names = list(results.keys())
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            labels1 = results[method1]['labels']
            labels2 = results[method2]['labels']
            
            chi2, p_value, contingency_table = chi_squared_test(labels1, labels2, method1, method2)
            print(f"\n{method1.upper()} vs {method2.upper()}:")
            print(f"  χ² = {chi2:.4f}")
            print(f"  p-value = {p_value:.4f}")
            print(f"  Significant (p < 0.05): {p_value < 0.05}")
            print(f"  Contingency table shape: {contingency_table.shape}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method, result in results.items():
        print(f"{method.upper()}: {result['n_clusters']} clusters")
        print(f"  Cluster sizes: {result['counts']}")

if __name__ == "__main__":
    main()
