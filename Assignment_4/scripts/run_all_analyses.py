"""Master script to run all Assignment 4 analyses."""
import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(result.stderr)
        return False
    else:
        print(f"SUCCESS: {description} completed")
        if result.stdout:
            print(result.stdout)
        return True

def main():
    parser = argparse.ArgumentParser(description='Run all Assignment 4 analyses')
    parser.add_argument('--skip-binary', action='store_true',
                       help='Skip binary classification models')
    parser.add_argument('--skip-cluster', action='store_true',
                       help='Skip cluster prediction models')
    parser.add_argument('--skip-variable-genes', action='store_true',
                       help='Skip variable gene number analysis')
    parser.add_argument('--skip-sample-auc', action='store_true',
                       help='Skip sample-specific AUC analysis')
    parser.add_argument('--skip-heatmap', action='store_true',
                       help='Skip signature heatmap')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    
    # Use python3 if available, otherwise use sys.executable
    import shutil
    python = shutil.which('python3') or sys.executable
    if python is None:
        python = 'python3'  # fallback
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Train binary classification models
    if not args.skip_binary:
        total_steps += 3
        models = [
            ('logistic_regression_model.py', 'Logistic Regression (Binary)'),
            ('random_forest_model.py', 'Random Forest (Binary)'),
            ('svm_model.py', 'SVM (Binary)')
        ]
        
        for model_script, model_name in models:
            cmd = [
                python,
                os.path.join(script_dir, model_script),
                '--label-type', 'binary'
            ]
            if run_command(cmd, f"Training {model_name}"):
                success_count += 1
    
    # Step 2: Train cluster prediction models
    if not args.skip_cluster:
        project_dir = os.path.dirname(assignment_dir)
        cluster_files = [
            ('kmeans_labels.tsv', 'K-means'),
            ('gmm_labels.tsv', 'GMM'),
            ('spectral_labels.tsv', 'Spectral')
        ]
        
        for cluster_file, cluster_name in cluster_files:
            cluster_path = os.path.join(project_dir, 'Assignment_3', 'results', cluster_file)
            if os.path.exists(cluster_path):
                total_steps += 3
                models = [
                    ('logistic_regression_model.py', 'Logistic Regression'),
                    ('random_forest_model.py', 'Random Forest'),
                    ('svm_model.py', 'SVM')
                ]
                
                for model_script, model_name in models:
                    cmd = [
                        python,
                        os.path.join(script_dir, model_script),
                        '--label-type', 'cluster',
                        '--cluster-file', cluster_path
                    ]
                    if run_command(cmd, f"Training {model_name} on {cluster_name} clusters"):
                        success_count += 1
    
    # Step 3: Variable gene number analysis
    if not args.skip_variable_genes:
        total_steps += 1
        cmd = [
            python,
            os.path.join(script_dir, 'variable_gene_analysis.py'),
            '--label-type', 'binary'
        ]
        if run_command(cmd, "Variable Gene Number Analysis"):
            success_count += 1
    
    # Step 4: Sample-specific AUC analysis
    if not args.skip_sample_auc:
        total_steps += 1
        cmd = [
            python,
            os.path.join(script_dir, 'sample_specific_auc_analysis.py')
        ]
        if run_command(cmd, "Sample-Specific AUC Analysis"):
            success_count += 1
    
    # Step 5: Signature heatmap
    if not args.skip_heatmap:
        total_steps += 1
        cmd = [
            python,
            os.path.join(script_dir, 'signature_heatmap.py')
        ]
        if run_command(cmd, "Signature Gene Heatmap"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("All analyses completed successfully!")
    else:
        print(f"Warning: {total_steps - success_count} step(s) failed. Check error messages above.")

if __name__ == "__main__":
    main()

