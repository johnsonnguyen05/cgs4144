"""Train models with different numbers of variable genes and compare performance."""
import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def subset_genes(input_file, n_genes, output_file):
    """Subset expression data to top N variable genes."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    subset_script = os.path.join(assignment_dir, 'subset_top_variable_genes.py')
    
    cmd = [
        sys.executable, subset_script,
        '--input', input_file,
        '--output', output_file,
        '--n', str(n_genes)
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Created subset with {n_genes} genes: {output_file}")

def train_model(script_name, input_file, output_dir, label_type='binary', cluster_file=None, metadata_file=None):
    """Train a model using the specified script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    cmd = [
        sys.executable, script_path,
        '--input', input_file,
        '--label-type', label_type,
        '--output-dir', output_dir
    ]
    
    if cluster_file:
        cmd.extend(['--cluster-file', cluster_file])
    
    # Note: metadata_file is not passed as argument since models use utils.py
    # which searches for it automatically
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error training {script_name}: {result.stderr}")
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}")  # Print first 500 chars
        return None
    
    # Read results file - try different possible names
    script_base = script_name.replace('_model.py', '').replace('.py', '')
    possible_results_files = [
        os.path.join(output_dir, f"{script_base}_results.txt"),
        os.path.join(output_dir, f"{script_base.replace('_', '_')}_results.txt"),
    ]
    
    for results_file in possible_results_files:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                content = f.read()
            return content
    
    return None

def extract_auc_from_results(results_content):
    """Extract AUC from results text."""
    for line in results_content.split('\n'):
        if 'AUC Score:' in line:
            try:
                auc_str = line.split('AUC Score:')[1].strip()
                if auc_str == 'None' or 'could not be computed' in auc_str:
                    return None
                return float(auc_str)
            except:
                return None
    return None

def main():
    parser = argparse.ArgumentParser(description='Analyze model performance with different gene numbers')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input expression data file (default: expression_data.tsv from Assignment_2)')
    parser.add_argument('--label-type', '-l', type=str, choices=['binary', 'cluster'],
                       default='binary', help='Type of labels to predict')
    parser.add_argument('--cluster-file', '-c', type=str, default=None,
                       help='Path to cluster labels file (required if label-type=cluster)')
    parser.add_argument('--models', '-m', nargs='+', 
                       default=['logistic_regression_model.py', 'random_forest_model.py', 'svm_model.py'],
                       help='Model scripts to run')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(assignment_dir)
    
    if args.input is None:
        # Try to find expression_data.tsv from Assignment 2
        expr_file_a2 = os.path.join(project_dir, 'Assignment_2', 'expression_data.tsv')
        if os.path.exists(expr_file_a2):
            args.input = expr_file_a2
        else:
            print(f"Error: Could not find expression_data.tsv. Please specify --input")
            sys.exit(1)
    
    gene_numbers = [10, 100, 1000, 10000]
    results_summary = []
    
    # Create temporary directory for subsets
    temp_dir = os.path.join(assignment_dir, 'temp_gene_subsets')
    os.makedirs(temp_dir, exist_ok=True)
    
    output_base = os.path.join(assignment_dir, 'results', 'variable_genes_analysis')
    os.makedirs(output_base, exist_ok=True)
    
    for n_genes in gene_numbers:
        print(f"\n{'='*60}")
        print(f"Analyzing models with {n_genes} genes")
        print(f"{'='*60}")
        
        # Create subset
        subset_file = os.path.join(temp_dir, f'expression_data_top{n_genes}.tsv')
        subset_genes(args.input, n_genes, subset_file)
        
        # Train each model
        for model_script in args.models:
            model_name = model_script.replace('_model.py', '').replace('.py', '')
            print(f"\nTraining {model_name} with {n_genes} genes...")
            
            output_dir = os.path.join(output_base, f'{model_name}_genes{n_genes}')
            os.makedirs(output_dir, exist_ok=True)
            
            results_content = train_model(
                model_script, 
                subset_file, 
                output_dir,
                label_type=args.label_type,
                cluster_file=args.cluster_file
            )
            
            if results_content:
                auc = extract_auc_from_results(results_content)
                results_summary.append({
                    'model': model_name,
                    'n_genes': n_genes,
                    'auc': auc
                })
                print(f"  AUC: {auc}")
            else:
                print(f"  Failed to extract results")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(output_base, 'auc_by_gene_number.tsv')
    results_df.to_csv(summary_file, sep='\t', index=False)
    print(f"\nSummary saved to {summary_file}")
    
    # Plot results
    if len(results_df) > 0:
        plt.figure(figsize=(10, 6))
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            plt.plot(model_data['n_genes'], model_data['auc'], 
                    marker='o', label=model, linewidth=2, markersize=8)
        
        plt.xlabel('Number of Genes', fontsize=12)
        plt.ylabel('AUC Score', fontsize=12)
        plt.title(f'Model Performance vs Number of Genes ({args.label_type} classification)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks(gene_numbers, gene_numbers)
        plt.tight_layout()
        
        plot_file = os.path.join(output_base, 'auc_vs_gene_number.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Cleanup temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary files in {temp_dir}")

if __name__ == "__main__":
    main()

