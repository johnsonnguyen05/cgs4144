"""Calculate sample-specific AUC and analyze model agreement."""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess_data
from logistic_regression_model import train_and_evaluate_lr
from random_forest_model import train_and_evaluate_rf
from svm_model import train_and_evaluate_svm

def get_all_predictions(models_config, input_file, label_type='binary', cluster_file=None):
    """
    Get predictions from all models for all samples.
    
    Returns:
    --------
    predictions_df : pd.DataFrame
        Samples x Models matrix with predicted class labels
    probabilities_dict : dict
        Dictionary mapping model names to probability arrays
    sample_names : np.ndarray
        Sample names/index
    label_encoders : dict
        Dictionary mapping model names to label encoders
    """
    X, y, le, sample_names = load_and_preprocess_data(
        input_file, 
        label_type=label_type,
        cluster_file=cluster_file
    )
    
    predictions_dict = {}
    probabilities_dict = {}
    label_encoders = {}
    
    for model_name, train_func in models_config.items():
        print(f"Training {model_name}...")
        results = train_func(X, y, return_full_predictions=True)
        
        # Get predictions for all samples
        y_pred_full = results['y_pred_full']
        y_prob_full = results['y_prob_full_data']
        
        # Map encoded predictions back to original labels
        pred_labels = le.inverse_transform(y_pred_full)
        predictions_dict[model_name] = pred_labels
        probabilities_dict[model_name] = y_prob_full
        label_encoders[model_name] = le
    
    # Create DataFrame
    predictions_df = pd.DataFrame(predictions_dict, index=sample_names)
    
    return predictions_df, probabilities_dict, sample_names, label_encoders

def calculate_sample_metrics(predictions_df, probabilities_dict, label_encoders):
    """
    Calculate per-sample metrics:
    - How many models predict each class
    - Prediction stability (variance/entropy)
    """
    sample_metrics = []
    
    for sample in predictions_df.index:
        sample_preds = predictions_df.loc[sample].values
        
        # Count predictions per class
        unique, counts = np.unique(sample_preds, return_counts=True)
        pred_counts = dict(zip(unique, counts))
        n_models = len(sample_preds)
        
        # Calculate prediction stability (entropy)
        # Higher entropy = less stable (more disagreement)
        # Lower entropy = more stable (more agreement)
        if len(unique) > 1:
            probs = counts / n_models
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0.0  # Perfect agreement
        
        # Calculate agreement (fraction of models agreeing on most common prediction)
        max_count = max(counts)
        agreement = max_count / n_models
        
        sample_metrics.append({
            'sample': sample,
            'n_models': n_models,
            'n_unique_predictions': len(unique),
            'entropy': entropy,
            'agreement': agreement,
            'most_common_prediction': unique[np.argmax(counts)],
            'most_common_count': max_count
        })
        
        # Add per-class counts
        all_classes = set()
        for le in label_encoders.values():
            all_classes.update(le.classes_)
        
        for class_label in all_classes:
            sample_metrics[-1][f'n_models_predict_{class_label}'] = pred_counts.get(class_label, 0)
    
    return pd.DataFrame(sample_metrics)

def calculate_stability_correlation(class_stability, cluster_stability):
    """
    Calculate correlation between class label prediction stability and cluster prediction stability.
    """
    # Merge on sample names
    merged = pd.merge(
        class_stability[['sample', 'entropy', 'agreement']],
        cluster_stability[['sample', 'entropy', 'agreement']],
        on='sample',
        suffixes=('_class', '_cluster')
    )
    
    # Calculate correlations
    entropy_corr, entropy_p = stats.pearsonr(merged['entropy_class'], merged['entropy_cluster'])
    agreement_corr, agreement_p = stats.pearsonr(merged['agreement_class'], merged['agreement_cluster'])
    
    return {
        'entropy_correlation': entropy_corr,
        'entropy_pvalue': entropy_p,
        'agreement_correlation': agreement_corr,
        'agreement_pvalue': agreement_p,
        'n_samples': len(merged)
    }

def main():
    parser = argparse.ArgumentParser(description='Sample-specific AUC and model agreement analysis')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input expression data file')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    
    if args.input is None:
        input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    else:
        input_file = args.input
    
    if args.output_dir is None:
        output_dir = os.path.join(assignment_dir, 'results', 'sample_specific_auc')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define models
    models_config = {
        'logistic_regression': train_and_evaluate_lr,
        'random_forest': train_and_evaluate_rf,
        'svm': train_and_evaluate_svm
    }
    
    print("="*60)
    print("ANALYZING BINARY CLASS PREDICTIONS (Trem2KO vs WT)")
    print("="*60)
    
    # Get predictions for binary classification
    binary_predictions, binary_probs, binary_samples, binary_label_encoders = get_all_predictions(
        models_config, input_file, label_type='binary'
    )
    
    # Calculate sample metrics for binary predictions
    binary_metrics = calculate_sample_metrics(
        binary_predictions, binary_probs, binary_label_encoders
    )
    
    # Save binary predictions
    binary_predictions.to_csv(os.path.join(output_dir, 'binary_predictions_matrix.tsv'), sep='\t')
    binary_metrics.to_csv(os.path.join(output_dir, 'binary_sample_metrics.tsv'), sep='\t', index=False)
    
    print("\nBinary prediction summary:")
    print(binary_metrics[['sample', 'most_common_prediction', 'agreement', 'entropy']].head(10))
    
    print("\n" + "="*60)
    print("ANALYZING CLUSTER PREDICTIONS")
    print("="*60)
    
    # Get cluster predictions from Assignment 3
    project_dir = os.path.dirname(assignment_dir)
    cluster_files = {
        'kmeans': os.path.join(project_dir, 'Assignment_3', 'results', 'kmeans_labels.tsv'),
        'gmm': os.path.join(project_dir, 'Assignment_3', 'results', 'gmm_labels.tsv'),
        'spectral': os.path.join(project_dir, 'Assignment_3', 'results', 'spectral_labels.tsv')
    }
    
    # Try each clustering result
    cluster_results = {}
    for cluster_name, cluster_file in cluster_files.items():
        if os.path.exists(cluster_file):
            print(f"\nAnalyzing {cluster_name} clusters...")
            try:
                cluster_predictions, cluster_probs, cluster_samples, cluster_label_encoders = get_all_predictions(
                    models_config, input_file, label_type='cluster', cluster_file=cluster_file
                )
                
                cluster_metrics = calculate_sample_metrics(
                    cluster_predictions, cluster_probs, cluster_label_encoders
                )
                
                cluster_results[cluster_name] = {
                    'predictions': cluster_predictions,
                    'metrics': cluster_metrics,
                    'samples': cluster_samples
                }
                
                # Save cluster predictions
                cluster_dir = os.path.join(output_dir, cluster_name)
                os.makedirs(cluster_dir, exist_ok=True)
                cluster_predictions.to_csv(
                    os.path.join(cluster_dir, f'{cluster_name}_predictions_matrix.tsv'), sep='\t'
                )
                cluster_metrics.to_csv(
                    os.path.join(cluster_dir, f'{cluster_name}_sample_metrics.tsv'), sep='\t', index=False
                )
                
            except Exception as e:
                print(f"Error analyzing {cluster_name}: {e}")
                continue
    
    # Calculate stability correlations
    print("\n" + "="*60)
    print("STABILITY CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_results = []
    
    for cluster_name, cluster_data in cluster_results.items():
        corr_results = calculate_stability_correlation(binary_metrics, cluster_data['metrics'])
        corr_results['cluster_method'] = cluster_name
        correlation_results.append(corr_results)
        
        print(f"\n{cluster_name.upper()} vs Binary Class:")
        print(f"  Entropy correlation: {corr_results['entropy_correlation']:.4f} (p={corr_results['entropy_pvalue']:.4e})")
        print(f"  Agreement correlation: {corr_results['agreement_correlation']:.4f} (p={corr_results['agreement_pvalue']:.4e})")
    
    # Multiple test correction
    if len(correlation_results) > 0:
        corr_df = pd.DataFrame(correlation_results)
        
        # Apply Benjamini-Hochberg correction
        pvalues = corr_df[['entropy_pvalue', 'agreement_pvalue']].values.flatten()
        _, pvalues_corrected, _, _ = multipletests(pvalues, method='fdr_bh')
        pvalues_corrected = pvalues_corrected.reshape(len(corr_df), 2)
        
        corr_df['entropy_pvalue_corrected'] = pvalues_corrected[:, 0]
        corr_df['agreement_pvalue_corrected'] = pvalues_corrected[:, 1]
        
        corr_df.to_csv(os.path.join(output_dir, 'stability_correlations.tsv'), sep='\t', index=False)
        
        print("\nAfter multiple test correction (Benjamini-Hochberg):")
        for _, row in corr_df.iterrows():
            print(f"\n{row['cluster_method'].upper()}:")
            print(f"  Entropy: p={row['entropy_pvalue_corrected']:.4e}")
            print(f"  Agreement: p={row['agreement_pvalue_corrected']:.4e}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Binary agreement distribution
    axes[0, 0].hist(binary_metrics['agreement'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Agreement (fraction of models agreeing)')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Binary Class Prediction Agreement')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binary entropy distribution
    axes[0, 1].hist(binary_metrics['entropy'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Entropy (prediction stability)')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Binary Class Prediction Stability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation plot (if we have cluster results)
    if len(cluster_results) > 0:
        cluster_name = list(cluster_results.keys())[0]
        cluster_metrics = cluster_results[cluster_name]['metrics']
        
        merged = pd.merge(
            binary_metrics[['sample', 'agreement']],
            cluster_metrics[['sample', 'agreement']],
            on='sample',
            suffixes=('_binary', '_cluster')
        )
        
        axes[1, 0].scatter(merged['agreement_binary'], merged['agreement_cluster'], alpha=0.6)
        axes[1, 0].set_xlabel('Binary Class Agreement')
        axes[1, 0].set_ylabel(f'{cluster_name.capitalize()} Cluster Agreement')
        axes[1, 0].set_title('Stability Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction counts
        pred_counts = binary_metrics['most_common_count'].value_counts().sort_index()
        axes[1, 1].bar(pred_counts.index, pred_counts.values, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Models Agreeing')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title('Model Agreement Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_specific_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

