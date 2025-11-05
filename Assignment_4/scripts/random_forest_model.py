import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def train_and_evaluate_rf(X, y):
    """Train Random Forest model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    # Prepare probabilities and AUC depending on binary vs multiclass
    y_prob_full = rf.predict_proba(X_test)
    num_classes = len(np.unique(y))

    if num_classes == 2:
        # Binary: take probability for class 1
        y_prob = y_prob_full[:, 1]
        from sklearn.preprocessing import label_binarize
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
    
    return {
        'model': rf,
        'accuracy': accuracy,
        'auc': auc,
        'conf_matrix': conf_matrix,
        'y_test': y_test,
        'y_prob': y_prob,
        'feature_importance': feature_importance
    }

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
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    assignment_dir = os.path.dirname(script_dir)  # Assignment_4/
    
    input_file = os.path.join(assignment_dir, 'expression_data_top5000.tsv')
    output_dir = os.path.join(assignment_dir, 'results/random_forest_model/')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data (returns scaled DataFrame, encoded labels, and LabelEncoder)
    X, y, le = load_and_preprocess_data(input_file)
    
    # Train and evaluate model
    results = train_and_evaluate_rf(X, y)
    
    # Save results
    with open(f'{output_dir}random_forest_results.txt', 'w') as f:
        f.write(f"Random Forest Model Results\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"AUC Score: {results['auc']:.4f}\n")
        else:
            f.write(f"AUC Score: None (multiclass or could not be computed)\n")
        f.write(f"Label classes: {list(le.classes_)}\n")
    
    # Plot and save figures
    plot_roc_curve(results['y_test'], results['y_prob'], 
                   f'{output_dir}random_forest_roc_curve.png')
    plot_confusion_matrix(results['conf_matrix'],
                         f'{output_dir}random_forest_confusion_matrix.png')
    plot_feature_importance(results['feature_importance'],
                          f'{output_dir}random_forest_feature_importance.png')

if __name__ == "__main__":
    main()