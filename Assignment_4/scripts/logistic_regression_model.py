import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
        metadata_file = os.path.join(project_dir, 'SRP119064', 'metadata_SRP119064.tsv')
    """Load and preprocess expression data.

    Improvements:
    - Reindex metadata to expression sample IDs and drop samples without labels.
    - Preserve feature names by returning a DataFrame for X (scaled).
    - Return the fitted LabelEncoder so callers can inspect classes if needed.
    """
    # Load expression data
    data = pd.read_csv(file_path, sep='\t', index_col=0)

    # Transpose the data so samples are rows and genes are columns (DataFrame)
    X = data.transpose()

    # Load metadata to get labels
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Extract subject information (genotype and age) as labels
    # The refinebio_subject column contains information about mouse type (wt vs trem2ko) and age (4m vs 8m)
    y_series = metadata.set_index('refinebio_accession_code')['refinebio_subject']

    # Align metadata to expression sample IDs; reindex will put NaN where missing
    y = y_series.reindex(X.index)

    # Drop samples with missing labels
    missing_mask = y.notna()
    if missing_mask.sum() != len(y):
        num_dropped = len(y) - int(missing_mask.sum())
        print(f"Warning: dropping {num_dropped} samples with missing metadata labels.")
        X = X.loc[missing_mask]
        y = y.loc[missing_mask]

    # Convert labels to categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features and keep column names
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, index=X.index, columns=X.columns)

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Labels: {le.classes_}")
    print(f"Label counts:\n{pd.Series(y).value_counts()}")

    return X_scaled, y_encoded, le


def train_and_evaluate_lr(X, y):
    """Train Logistic Regression model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_test)
    y_prob_full = lr.predict_proba(X_test)

    # Determine binary vs multiclass from fitted model
    num_classes = len(lr.classes_)
    if num_classes == 2:
        y_prob = y_prob_full[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None
    else:
        # multiclass: compute macro AUC using one-vs-rest
        from sklearn.preprocessing import label_binarize
        try:
            y_test_binarized = label_binarize(y_test, classes=lr.classes_)
            auc = roc_auc_score(y_test_binarized, y_prob_full, average='macro', multi_class='ovr')
        except Exception:
            auc = None
        y_prob = y_prob_full

    # Calculate metrics
    accuracy = lr.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Get feature coefficients: align with column names. For multiclass, aggregate absolute coeffs
    try:
        if num_classes == 2:
            feature_coef = pd.Series(lr.coef_[0], index=X.columns)
        else:
            coef_abs_sum = np.abs(lr.coef_).sum(axis=0)
            feature_coef = pd.Series(coef_abs_sum, index=X.columns)
    except Exception:
        # fallback
        feature_coef = pd.Series(np.ravel(lr.coef_), index=X.columns[:np.ravel(lr.coef_).shape[0]])
    
    return {
        'model': lr,
        'accuracy': accuracy,
        'auc': auc,
        'conf_matrix': conf_matrix,
        'y_test': y_test,
        'y_prob': y_prob,
        'feature_coef': feature_coef
    }

def plot_roc_curve(y_test, y_prob, output_path):
    """Plot and save ROC curve.

    Only supports binary (1D y_prob). For multiclass, this will skip plotting.
    """
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
    plt.title('ROC Curve - Logistic Regression')
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(output_path)
    plt.close()

def plot_feature_coefficients(feature_coef, output_path):
    """Plot and save feature coefficients."""
    plt.figure(figsize=(10, 6))
    feature_coef.abs().sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Most Important Features - Logistic Regression')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    assignment_dir = os.path.dirname(script_dir)  # Assignment_4/
    
    input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    output_dir = os.path.join(assignment_dir, 'results/logistic_regression_model/')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    # load returns DataFrame X (scaled), encoded y, and LabelEncoder
    X, y, le = load_and_preprocess_data(input_file)
    
    # Train and evaluate model
    results = train_and_evaluate_lr(X, y)
    
    # Save results
    with open(f'{output_dir}logistic_regression_results.txt', 'w') as f:
        f.write(f"Logistic Regression Model Results\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"AUC Score: {results['auc']:.4f}\n")
        else:
            f.write(f"AUC Score: None (multiclass or could not be computed)\n")
        f.write(f"Label classes: {list(le.classes_)}\n")
    
    # Plot and save figures
    plot_roc_curve(results['y_test'], results['y_prob'], 
                  os.path.join(output_dir, 'logistic_regression_roc_curve.png'))
    plot_confusion_matrix(results['conf_matrix'],
                         os.path.join(output_dir, 'logistic_regression_confusion_matrix.png'))
    plot_feature_coefficients(results['feature_coef'],
                            os.path.join(output_dir, 'logistic_regression_feature_importance.png'))

if __name__ == "__main__":
    main()