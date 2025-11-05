import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess the expression data."""
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    # Assume the first column contains the labels (modify as needed)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_and_evaluate_lr(X, y):
    """Train Logistic Regression model and evaluate its performance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = lr.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature coefficients
    feature_coef = pd.Series(lr.coef_[0])
    
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
    """Plot and save ROC curve."""
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
    input_file = '../expression_data_top5000.tsv'
    output_dir = '../results/'
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(input_file)
    
    # Train and evaluate model
    results = train_and_evaluate_lr(X, y)
    
    # Save results
    with open(f'{output_dir}logistic_regression_results.txt', 'w') as f:
        f.write(f"Logistic Regression Model Results\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"AUC Score: {results['auc']:.4f}\n")
    
    # Plot and save figures
    plot_roc_curve(results['y_test'], results['y_prob'], 
                   f'{output_dir}logistic_regression_roc_curve.png')
    plot_confusion_matrix(results['conf_matrix'],
                         f'{output_dir}logistic_regression_confusion_matrix.png')
    plot_feature_coefficients(results['feature_coef'],
                            f'{output_dir}logistic_regression_feature_importance.png')

if __name__ == "__main__":
    main()