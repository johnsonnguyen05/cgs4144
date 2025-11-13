"""Create heatmap with signature genes from predictive models and Assignment 1 annotations."""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from utils import get_metadata_groups

def extract_signature_genes(model_results_dir, top_n=50):
    """
    Extract top N signature genes from model feature importance/coefficients.
    
    Returns:
    --------
    signature_genes : set
        Set of gene names that are in the signature
    """
    signature_genes = set()
    
    # Try to find feature importance/coefficient files
    feature_files = [
        'feature_coefficients.tsv',
        'feature_importance.tsv'
    ]
    
    for feature_file in feature_files:
        file_path = os.path.join(model_results_dir, feature_file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                # Get top N genes
                top_genes = df.head(top_n).index.tolist()
                signature_genes.update(top_genes)
                print(f"  Extracted {len(top_genes)} genes from {feature_file}")
                break
            except Exception as e:
                print(f"  Error reading {feature_file}: {e}")
                continue
    
    return signature_genes

def get_all_signature_genes(results_base_dir, top_n_per_model=50):
    """
    Extract signature genes from all models.
    
    Returns:
    --------
    all_signature_genes : set
        Union of all signature genes across models
    """
    all_signature_genes = set()
    
    model_dirs = [
        'logistic_regression_model',
        'random_forest_model',
        'svm'
    ]
    
    print("Extracting signature genes from models:")
    for model_dir in model_dirs:
        model_path = os.path.join(results_base_dir, model_dir)
        if os.path.exists(model_path):
            print(f"  {model_dir}:")
            sig_genes = extract_signature_genes(model_path, top_n=top_n_per_model)
            all_signature_genes.update(sig_genes)
            print(f"    Found {len(sig_genes)} signature genes")
    
    print(f"\nTotal unique signature genes: {len(all_signature_genes)}")
    return all_signature_genes

def create_annotation_sidebar(sample_names, metadata_file=None):
    """
    Create annotation sidebar for Assignment 1 groups (Trem2KO vs WT).
    
    Returns:
    --------
    annotation_df : pd.DataFrame
        DataFrame with sample names and group annotations
    """
    groups = get_metadata_groups(metadata_file)
    
    # Align to sample names
    annotation_series = groups.reindex(sample_names)
    
    # Create DataFrame
    annotation_df = pd.DataFrame({
        'Group': annotation_series.values
    }, index=sample_names)
    
    # Map to colors
    color_map = {
        'Trem2KO': '#FF6B6B',  # Red
        'WT': '#4ECDC4',        # Teal
        'Unknown': '#95A5A6'    # Gray
    }
    
    annotation_colors = annotation_df['Group'].map(color_map).values
    
    return annotation_df, annotation_colors, color_map

def create_signature_heatmap(expression_data, signature_genes, annotation_colors, 
                             color_map, output_path, sample_names, has_annotations=True):
    """
    Create heatmap with signature genes, annotations, and dendrograms.
    """
    # Subset to signature genes
    available_genes = [g for g in signature_genes if g in expression_data.columns]
    expr_subset = expression_data[available_genes]
    
    print(f"Using {len(available_genes)} signature genes (out of {len(signature_genes)} requested)")
    
    # Scale the data
    scaler = StandardScaler()
    expr_scaled = pd.DataFrame(
        scaler.fit_transform(expr_subset),
        index=expr_subset.index,
        columns=expr_subset.columns
    )
    
    # Prepare clustermap arguments
    clustermap_kwargs = {
        'data': expr_scaled.T,  # Transpose: genes as rows, samples as columns
        'cmap': 'RdBu_r',
        'center': 0,
        'figsize': (15, max(10, len(available_genes) * 0.1)),
        'cbar_kws': {'label': 'Scaled Expression'},
        'xticklabels': False,  # Don't show all sample names (too many)
        'yticklabels': True,   # Show gene names
        'dendrogram_ratio': (0.1, 0.2),  # Ratio for row and col dendrograms
        'cbar_pos': (0.02, 0.8, 0.03, 0.15)
    }
    
    # Add annotations if available
    if has_annotations and annotation_colors is not None:
        annotation_df = pd.DataFrame({
            'Group': pd.Categorical(
                [color_map.get(c, 'Unknown') for c in annotation_colors],
                categories=['Trem2KO', 'WT', 'Unknown']
            )
        }, index=sample_names)
        
        group_colors = {
            'Trem2KO': '#FF6B6B',
            'WT': '#4ECDC4',
            'Unknown': '#95A5A6'
        }
        
        clustermap_kwargs['col_colors'] = annotation_df['Group'].map(group_colors)
    
    # Create clustermap
    g = sns.clustermap(**clustermap_kwargs)
    
    # Add title
    if has_annotations:
        title = 'Gene Expression Heatmap: Predictive Model Signatures\nAnnotated with Assignment 1 Groups (Trem2KO vs WT)'
    else:
        title = 'Gene Expression Heatmap: Predictive Model Signatures'
    g.fig.suptitle(title, fontsize=14, y=0.98)
    
    # Add legend for annotations if available
    if has_annotations and annotation_colors is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=group_colors['Trem2KO'], label='Trem2KO'),
            Patch(facecolor=group_colors['WT'], label='WT'),
            Patch(facecolor=group_colors['Unknown'], label='Unknown')
        ]
        g.ax_col_dendrogram.legend(handles=legend_elements, loc='upper left', 
                                  bbox_to_anchor=(1.02, 1), frameon=True)
    
    # Set axis labels
    g.ax_heatmap.set_xlabel('Samples', fontsize=12)
    g.ax_heatmap.set_ylabel('Genes (Signature from Predictive Models)', fontsize=12)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create signature gene heatmap with annotations')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input expression data file')
    parser.add_argument('--results-dir', '-r', type=str, default=None,
                       help='Base directory containing model results')
    parser.add_argument('--top-n', '-n', type=int, default=50,
                       help='Number of top genes per model to include (default: 50)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    
    if args.input is None:
        input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    else:
        input_file = args.input
    
    if args.results_dir is None:
        results_base_dir = os.path.join(assignment_dir, 'results')
    else:
        results_base_dir = args.results_dir
    
    if args.output is None:
        output_file = os.path.join(assignment_dir, 'results', 'visualization', 'signature_heatmap.png')
    else:
        output_file = args.output
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load expression data
    print("Loading expression data...")
    expr_data = pd.read_csv(input_file, sep='\t', index_col=0)
    expr_data = expr_data.T  # Transpose: samples x genes
    print(f"Expression data shape: {expr_data.shape}")
    
    # Extract signature genes
    print("\nExtracting signature genes...")
    signature_genes = get_all_signature_genes(results_base_dir, top_n_per_model=args.top_n)
    
    if len(signature_genes) == 0:
        print("Warning: No signature genes found. Using top 100 most variable genes instead.")
        # Fallback: use top variable genes
        variances = expr_data.var(axis=0)
        signature_genes = set(variances.nlargest(100).index)
    
    # Create annotations
    print("\nCreating annotations...")
    sample_names = expr_data.index.values
    
    try:
        annotation_df, annotation_colors, color_map = create_annotation_sidebar(sample_names)
        print(f"Annotation groups: {annotation_df['Group'].value_counts().to_dict()}")
        has_annotations = True
    except FileNotFoundError as e:
        print(f"Warning: Could not load metadata for annotations: {e}")
        print("Creating heatmap without Assignment 1 group annotations...")
        annotation_df = None
        annotation_colors = None
        color_map = None
        has_annotations = False
    
    # Create heatmap
    print("\nCreating heatmap...")
    create_signature_heatmap(
        expr_data, 
        signature_genes, 
        annotation_colors,
        color_map,
        output_file,
        sample_names,
        has_annotations=has_annotations
    )
    
    # Save signature gene list
    sig_file = os.path.join(os.path.dirname(output_file), 'signature_genes.txt')
    with open(sig_file, 'w') as f:
        for gene in sorted(signature_genes):
            f.write(f"{gene}\n")
    print(f"Signature gene list saved to {sig_file}")
    
    print(f"\nCompleted! Heatmap saved to {output_file}")

if __name__ == "__main__":
    main()

