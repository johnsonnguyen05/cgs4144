"""Utility functions for Assignment 4 supervised analysis."""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def assign_binary_group(x):
    """Extract binary Trem2KO vs WT from refinebio_subject string."""
    x = str(x).lower()
    if "trem2ko" in x or "trem2 ko" in x:
        return "Trem2KO"
    elif "wt" in x:
        return "WT"
    else:
        return "Unknown"


def load_and_preprocess_data(file_path, metadata_file=None, label_type='binary', cluster_file=None):
    """
    Load and preprocess expression data with flexible label options.
    
    Parameters:
    -----------
    file_path : str
        Path to expression data TSV (genes x samples)
    metadata_file : str, optional
        Path to metadata file. If None, searches in SRP119064/
    label_type : str
        'binary' for Trem2KO vs WT, 'multiclass' for full refinebio_subject, 'cluster' for Assignment 3 clusters
    cluster_file : str, optional
        Path to cluster labels TSV (required if label_type='cluster')
    
    Returns:
    --------
    X_scaled : pd.DataFrame
        Scaled expression data (samples x genes)
    y : np.ndarray
        Encoded labels
    le : LabelEncoder
        Fitted label encoder
    sample_names : np.ndarray
        Sample names/index
    """
    if metadata_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        
        # Try multiple possible locations for metadata
        possible_paths = [
            os.path.join(project_dir, 'SRP119064', 'metadata_SRP119064.tsv'),
            os.path.join(project_dir, 'Assignment_2', 'SRP119064', 'metadata_SRP119064.tsv'),
            os.path.join(os.path.dirname(project_dir), 'SRP119064', 'metadata_SRP119064.tsv'),
        ]
        
        metadata_file = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_file = path
                break
        
        if metadata_file is None and label_type != 'cluster':
            raise FileNotFoundError(
                f"Metadata file not found. Please provide metadata file or place it at one of:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nOr provide --metadata-file argument."
            )
    
    # Load expression data
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    X = data.transpose()  # samples x genes
    
    # Load labels based on label_type
    if label_type == 'cluster':
        if cluster_file is None:
            raise ValueError("cluster_file must be provided when label_type='cluster'")
        cluster_df = pd.read_csv(cluster_file, sep='\t', index_col=0)
        # Align cluster labels to expression samples
        y_series = cluster_df['cluster'].reindex(X.index)
        y_labels = y_series.dropna()
        X = X.loc[y_labels.index]
        y_raw = y_labels.values.astype(str)
    else:
        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        metadata = pd.read_csv(metadata_file, sep='\t')
        y_series = metadata.set_index('refinebio_accession_code')['refinebio_subject']
        y_series = y_series.reindex(X.index)
        
        # Drop samples with missing labels
        missing_mask = y_series.notna()
        X = X.loc[missing_mask]
        y_raw_series = y_series.loc[missing_mask]
        
        if label_type == 'binary':
            # Convert to binary Trem2KO vs WT
            y_raw = y_raw_series.map(assign_binary_group).values
            # Filter out "Unknown" samples
            valid_mask = y_raw != "Unknown"
            X = X.loc[valid_mask]
            y_raw = y_raw[valid_mask]
        else:  # multiclass
            y_raw = y_raw_series.values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, index=X.index, columns=X.columns)
    
    print(f"Loaded {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features.")
    print(f"Labels: {le.classes_}")
    print(f"Label counts:\n{pd.Series(y_raw).value_counts()}")
    
    return X_scaled, y_encoded, le, X_scaled.index.values


def get_metadata_groups(metadata_file=None):
    """Load Assignment 1 binary groups (Trem2KO vs WT) for visualization."""
    if metadata_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        
        # Try multiple possible locations for metadata
        possible_paths = [
            os.path.join(project_dir, 'SRP119064', 'metadata_SRP119064.tsv'),
            os.path.join(project_dir, 'Assignment_2', 'SRP119064', 'metadata_SRP119064.tsv'),
            os.path.join(os.path.dirname(project_dir), 'SRP119064', 'metadata_SRP119064.tsv'),
        ]
        
        metadata_file = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_file = path
                break
        
        if metadata_file is None:
            raise FileNotFoundError(
                f"Metadata file not found. Please provide metadata file or place it at one of:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = pd.read_csv(metadata_file, sep='\t')
    metadata = metadata.set_index('refinebio_accession_code')
    groups = metadata['refinebio_subject'].map(assign_binary_group)
    return groups

