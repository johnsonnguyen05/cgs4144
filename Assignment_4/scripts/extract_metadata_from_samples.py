"""Extract metadata from sample names and expression data patterns.
This is a fallback if metadata file is not available."""
import os
import pandas as pd
import numpy as np

def infer_metadata_from_expression(expression_file):
    """
    Try to infer Trem2KO vs WT from expression patterns.
    This is a fallback method - ideally use the actual metadata file.
    """
    # Load expression data
    expr = pd.read_csv(expression_file, sep='\t', index_col=0)
    expr = expr.T  # samples x genes
    
    # Check if we can use Trem2 expression as a proxy
    # Trem2KO should have lower Trem2 expression
    if 'Trem2' in expr.columns:
        trem2_expr = expr['Trem2']
        # Use median split - samples below median are likely Trem2KO
        median_trem2 = trem2_expr.median()
        groups = trem2_expr.apply(lambda x: 'Trem2KO' if x < median_trem2 else 'WT')
        
        print(f"Using Trem2 expression as proxy (median={median_trem2:.2f})")
        print(f"Groups: Trem2KO={sum(groups=='Trem2KO')}, WT={sum(groups=='WT')}")
        
        # Create metadata-like structure
        metadata = pd.DataFrame({
            'refinebio_accession_code': groups.index,
            'refinebio_subject': groups.values
        })
        
        return metadata
    else:
        raise ValueError("Cannot infer groups: Trem2 gene not found in expression data")

def create_metadata_file(expression_file, output_path):
    """Create a metadata file from expression data patterns."""
    metadata = infer_metadata_from_expression(expression_file)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metadata
    metadata.to_csv(output_path, sep='\t', index=False)
    print(f"Created metadata file at: {output_path}")
    print("\nWARNING: This is inferred from expression patterns, not actual metadata.")
    print("For best results, use the actual metadata file from SRP119064.")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(assignment_dir)
    
    # Try to find expression data
    expr_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    if not os.path.exists(expr_file):
        expr_file = os.path.join(project_dir, 'Assignment_2', 'expression_data.tsv')
    
    if not os.path.exists(expr_file):
        print(f"Error: Could not find expression data file")
        sys.exit(1)
    
    # Create metadata file
    metadata_path = os.path.join(project_dir, 'SRP119064', 'metadata_SRP119064.tsv')
    create_metadata_file(expr_file, metadata_path)

