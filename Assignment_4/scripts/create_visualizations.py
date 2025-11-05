import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the expression data."""
    return pd.read_csv(file_path, sep='\t', index_col=0)

def create_heatmap(data, output_path):
    """Create and save heatmap with sample annotations."""
    plt.figure(figsize=(15, 10))
    
    # Create the heatmap
    sns.clustermap(data, 
                  cmap='RdBu_r',
                  xticklabels=True,
                  yticklabels=True,
                  figsize=(15, 10))
    
    plt.title('Gene Expression Heatmap with Hierarchical Clustering')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_dendrogram(data, output_path):
    """Create and save dendrogram."""
    plt.figure(figsize=(10, 7))
    
    # Calculate linkage
    linkage_matrix = linkage(data.T, method='ward')
    
    # Create dendrogram
    dendrogram(linkage_matrix)
    
    plt.title('Sample Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    assignment_dir = os.path.dirname(script_dir)  # Assignment_4/
    
    input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    output_dir = os.path.join(assignment_dir, 'results/visualization/')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data(input_file)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data),
        index=data.index,
        columns=data.columns
    )
    
    # Create visualizations
    create_heatmap(scaled_data, os.path.join(output_dir, 'heatmap.png'))
    create_dendrogram(scaled_data, os.path.join(output_dir, 'dendrogram.png'))

if __name__ == "__main__":
    main()