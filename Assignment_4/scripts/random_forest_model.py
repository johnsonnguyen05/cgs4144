import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess_data

def train_and_evaluate_rf(X, y, return_full_predictions=False):
    """Train Random Forest model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = rf.predict(X_test)
    y_prob_full = rf.predict_proba(X_test)
    
    # Also get predictions on full dataset if requested
    y_pred_full = None
    y_prob_full_data = None
    if return_full_predictions:
        y_pred_full = rf.predict(X)
        y_prob_full_data = rf.predict_proba(X)
    
    # Prepare probabilities and AUC depending on binary vs multiclass
    num_classes = len(np.unique(y))

    if num_classes == 2:
        # Binary: take probability for class 1
        y_prob = y_prob_full[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        # Multiclass: compute macro-average AUC using one-vs-rest
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y)
        y_test_binarized = label_binarize(y_test, classes=classes)
        try:
            auc = roc_auc_score(y_test_binarized, y_prob_full, average='macro', multi_class='ovr')
        except Exception:
            auc = None
        y_prob = y_prob_full

    # Calculate metrics
    accuracy = rf.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importance (label with feature names)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
    
    result = {
        'model': rf,
        'accuracy': accuracy,
        'auc': auc,
        'conf_matrix': conf_matrix,
        'y_test': y_test,
        'y_prob': y_prob,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'X_train': X_train
    }
    
    if return_full_predictions:
        result['y_pred_full'] = y_pred_full
        result['y_prob_full_data'] = y_prob_full_data
    
    return result

def plot_roc_curve(y_test, y_prob, output_path):
    """Plot and save ROC curve.

    This function only plots a ROC curve for binary classification (1D y_prob).
    For multiclass, the caller should skip or implement per-class curves.
    """
    # If y_prob is 2D (multiclass), skip plotting here.
    if y_prob is None:
        print("Skipping ROC plot because AUC/probabilities are not available.")
        return
    if isinstance(y_prob, np.ndarray) and y_prob.ndim == 2:
        print("Multiclass probabilities detected; skipping single ROC plot.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_importance, output_path):
    """Plot and save feature importance."""
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Most Important Features - Random Forest')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Random Forest model')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input expression data file (default: expression_data_top5000.tsv)')
    parser.add_argument('--label-type', '-l', type=str, choices=['binary', 'multiclass', 'cluster'],
                       default='binary', help='Type of labels to predict (default: binary)')
    parser.add_argument('--cluster-file', '-c', type=str, default=None,
                       help='Path to cluster labels file (required if label-type=cluster)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (default: results/random_forest_model/)')
    
    args = parser.parse_args()
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_dir = os.path.dirname(script_dir)
    
    if args.input is None:
        input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    else:
        input_file = args.input
    
    if args.output_dir is None:
        label_suffix = f"_{args.label_type}" if args.label_type != 'binary' else ''
        output_dir = os.path.join(assignment_dir, f'results/random_forest_model{label_suffix}/')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    X, y, le, sample_names = load_and_preprocess_data(
        input_file, 
        label_type=args.label_type,
        cluster_file=args.cluster_file
    )
    
    # Train and evaluate model
    results = train_and_evaluate_rf(X, y)
    
    # Save results
    results_file = os.path.join(output_dir, 'random_forest_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Random Forest Model Results\n")
        f.write(f"Label Type: {args.label_type}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"AUC Score: {results['auc']:.4f}\n")
        else:
            f.write(f"AUC Score: None (multiclass or could not be computed)\n")
        f.write(f"Label classes: {list(le.classes_)}\n")
        f.write(f"Number of classes: {len(le.classes_)}\n")
    
    # Plot and save figures
    plot_roc_curve(results['y_test'], results['y_prob'], 
                   os.path.join(output_dir, 'random_forest_roc_curve.png'))
    plot_confusion_matrix(results['conf_matrix'],
                         os.path.join(output_dir, 'random_forest_confusion_matrix.png'))
    plot_feature_importance(results['feature_importance'],
                          os.path.join(output_dir, 'random_forest_feature_importance.png'))
    
    # Save feature importance for signature extraction
    feature_file = os.path.join(output_dir, 'feature_importance.tsv')
    results['feature_importance'].sort_values(ascending=False).to_csv(
        feature_file, sep='\t', header=['importance']
    )
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()