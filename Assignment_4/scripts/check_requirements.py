"""Check if all required files exist and provide helpful error messages."""
import os
import sys
import pandas as pd

def check_metadata_file():
    """Check if metadata file exists in common locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    
    possible_paths = [
        os.path.join(project_dir, 'SRP119064', 'metadata_SRP119064.tsv'),
        os.path.join(project_dir, 'Assignment_2', 'SRP119064', 'metadata_SRP119064.tsv'),
        os.path.join(os.path.dirname(project_dir), 'SRP119064', 'metadata_SRP119064.tsv'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path, True
    
    return None, False

def main():
    print("Checking requirements for Assignment 4...")
    print("="*60)
    
    # Check metadata
    metadata_path, metadata_exists = check_metadata_file()
    if metadata_exists:
        print(f"✓ Metadata file found: {metadata_path}")
        
        # Check if it's inferred (has simple structure)
        try:
            df = pd.read_csv(metadata_path, sep='\t', nrows=5)
            if 'refinebio_subject' in df.columns and df['refinebio_subject'].nunique() == 2:
                print("  Note: This appears to be inferred from expression data (see extract_metadata_from_samples.py)")
        except:
            pass
    else:
        print("✗ Metadata file NOT found")
        print("\n  The metadata file is required for binary classification.")
        print("  Expected location: SRP119064/metadata_SRP119064.tsv")
        print("\n  You can:")
        print("  1. Generate metadata from expression patterns:")
        print("     python Assignment_4/scripts/extract_metadata_from_samples.py")
        print("  2. Place the actual metadata file at: /Users/r/Desktop/Projects/cgs4144/SRP119064/metadata_SRP119064.tsv")
        print("  3. Or skip binary classification and only run cluster predictions:")
        print("     ./run_analysis.sh --skip-binary")
        print("\n  Note: Cluster predictions work without metadata file.")
    
    # Check expression data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    expr_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    if os.path.exists(expr_file):
        print(f"✓ Expression data found: {expr_file}")
    else:
        print(f"✗ Expression data NOT found: {expr_file}")
    
    # Check cluster files
    project_dir = os.path.dirname(assignment_dir)
    cluster_files = [
        ('kmeans', os.path.join(project_dir, 'Assignment_3', 'results', 'kmeans_labels.tsv')),
        ('GMM', os.path.join(project_dir, 'Assignment_3', 'results', 'gmm_labels.tsv')),
        ('Spectral', os.path.join(project_dir, 'Assignment_3', 'results', 'spectral_labels.tsv')),
    ]
    
    print("\nCluster label files:")
    for name, path in cluster_files:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT found at {path}")
    
    print("\n" + "="*60)
    if metadata_exists:
        print("All requirements met! You can run all analyses.")
    else:
        print("You can run cluster predictions (--skip-binary) but binary classification requires metadata file.")

if __name__ == "__main__":
    main()

