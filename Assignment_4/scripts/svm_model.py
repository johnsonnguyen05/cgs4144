import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess_data


def train_and_evaluate_svm(X, y, return_full_predictions=False):
    """Train SVM model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train SVM model
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)
    
    # Also get predictions on full dataset if requested
    y_pred_full = None
    y_prob_full_data = None
    if return_full_predictions:
        y_pred_full = svm.predict(X)
        y_prob_full_data = svm.predict_proba(X)
    
    # Calculate metrics
    accuracy = svm.score(X_test, y_test)
    
    # For multi-class ROC AUC, we use the 'ovr' (one-vs-rest) approach
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = None
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    result = {
        'model': svm,
        'accuracy': accuracy,
        'auc': auc,
        'conf_matrix': conf_matrix,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classes': svm.classes_,
        'X_test': X_test,
        'X_train': X_train
    }
    
    if return_full_predictions:
        result['y_pred_full'] = y_pred_full
        result['y_prob_full_data'] = y_prob_full_data
    
    return result

def plot_roc_curves(y_test, y_prob, classes, output_path):
    """Plot and save ROC curves for multi-class classification."""
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    # Binarize the labels for ROC curve calculation
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_bin[:, i], y_prob[:, i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot individual class ROC curves
    for i, label in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Support Vector Machine (One-vs-Rest)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, classes, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Support Vector Machine')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train SVM model')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input expression data file (default: expression_data_top5000.tsv)')
    parser.add_argument('--label-type', '-l', type=str, choices=['binary', 'multiclass', 'cluster'],
                       default='binary', help='Type of labels to predict (default: binary)')
    parser.add_argument('--cluster-file', '-c', type=str, default=None,
                       help='Path to cluster labels file (required if label-type=cluster)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (default: results/svm/)')
    
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
        output_dir = os.path.join(assignment_dir, f'results/svm{label_suffix}/')
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
    results = train_and_evaluate_svm(X, y)
    
    # Save results
    results_path = os.path.join(output_dir, 'svm_results.txt')
    roc_path = os.path.join(output_dir, 'svm_roc_curves.png')
    conf_matrix_path = os.path.join(output_dir, 'svm_confusion_matrix.png')
    
    with open(results_path, 'w') as f:
        f.write(f"SVM Model Results\n")
        f.write(f"Label Type: {args.label_type}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"AUC Score: {results['auc']:.4f}\n")
        else:
            f.write(f"AUC Score: None (could not be computed)\n")
        f.write(f"Label classes: {list(le.classes_)}\n")
        f.write(f"Number of classes: {len(le.classes_)}\n")
    
    # Plot and save figures
    plot_roc_curves(results['y_test'], results['y_prob'], results['classes'], roc_path)
    plot_confusion_matrix(results['conf_matrix'], results['classes'], conf_matrix_path)
    
    # For SVM, we can't easily extract feature importance, but we can save support vectors info
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()