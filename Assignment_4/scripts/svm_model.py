import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path, metadata_file=None):
    if metadata_file is None:
        # Get the base directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
        project_dir = os.path.dirname(os.path.dirname(script_dir))  # cgs4144/
        metadata_file = os.path.join(project_dir, 'refinebio', 'SRP119064', 'metadata_SRP119064.tsv')
    """Load and preprocess expression data."""
    # Load expression data
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Transpose the data so samples are rows and genes are columns
    X = data.transpose()
    
    # Load metadata to get labels
    metadata = pd.read_csv(metadata_file, sep='\t')
    
    # Extract subject information (genotype and age) as labels
    # The refinebio_subject column contains information about mouse type (wt vs trem2ko) and age (4m vs 8m)
    y = metadata.set_index('refinebio_accession_code')['refinebio_subject']
    
    # Make sure X and y are aligned
    y = y[X.index]
    
    # Convert labels to categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Labels: {le.classes_}")
    print(f"Label counts: {pd.Series(y).value_counts()}")
    
    return X_scaled, y_encoded


def train_and_evaluate_svm(X, y):
    """Train SVM model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train SVM model
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = svm.score(X_test, y_test)
    
    # For multi-class ROC AUC, we use the 'ovr' (one-vs-rest) approach
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'model': svm,
        'accuracy': accuracy,
        'auc': auc,
        'conf_matrix': conf_matrix,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classes': svm.classes_
    }

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
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    assignment_dir = os.path.dirname(script_dir)  # Assignment_4/
    
    input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    output_dir = os.path.join(assignment_dir, 'results/svm/')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(input_file)
    
    # Train and evaluate model
    results = train_and_evaluate_svm(X, y)
    
    # Save results
    results_path = os.path.join(output_dir, 'svm_results.txt')
    roc_path = os.path.join(output_dir, 'svm_roc_curves.png')
    conf_matrix_path = os.path.join(output_dir, 'svm_confusion_matrix.png')
    
    with open(results_path, 'w') as f:
        f.write(f"SVM Model Results\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"AUC Score: {results['auc']:.4f}\n")
    
    # Plot and save figures
    plot_roc_curves(results['y_test'], results['y_prob'], results['classes'], roc_path)
    plot_confusion_matrix(results['conf_matrix'], results['classes'], conf_matrix_path)

if __name__ == "__main__":
    main()