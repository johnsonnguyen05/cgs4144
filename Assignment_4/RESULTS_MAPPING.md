# Results File Mapping to Assignment Instructions

This document maps each instruction from the assignment to the corresponding result files.

## 1. Subset to 5,000 Most Variable Genes

**Instruction:** Subset your data to the 5,000 most variable genes

**Result Files:**
- `expression_data_top5000.tsv` (input file for all models)
- `subset_top_variable_genes.py` (script that created it)

---

## 2. Predict Two Groups from Assignment 1 (Trem2KO vs WT)

**Instruction:** Have the algorithms predict the two groups from assignment 1 (e.g. tumor vs normal)

**Result Files (for each model):**

### Logistic Regression:
- `results/logistic_regression_model/logistic_regression_results.txt` - Accuracy, AUC, label classes
- `results/logistic_regression_model/logistic_regression_roc_curve.png` - ROC curve
- `results/logistic_regression_model/logistic_regression_confusion_matrix.png` - Confusion matrix
- `results/logistic_regression_model/logistic_regression_feature_importance.png` - Top 20 features
- `results/logistic_regression_model/feature_coefficients.tsv` - All feature coefficients

### Random Forest:
- `results/random_forest_model/random_forest_results.txt` - Accuracy, AUC, label classes
- `results/random_forest_model/random_forest_roc_curve.png` - ROC curve
- `results/random_forest_model/random_forest_confusion_matrix.png` - Confusion matrix
- `results/random_forest_model/random_forest_feature_importance.png` - Top 20 features
- `results/random_forest_model/feature_importance.tsv` - All feature importances

### SVM:
- `results/svm/svm_results.txt` - Accuracy, AUC, label classes
- `results/svm/svm_roc_curves.png` - ROC curves (multiclass)
- `results/svm/svm_confusion_matrix.png` - Confusion matrix

---

## 3. Retrain to Predict Clusters from Assignment 3

**Instruction:** Retrain the algorithms to predict the clusters from assignment 3

**Result Files (for each model + cluster method):**

### Logistic Regression on Clusters:
- `results/logistic_regression_model_cluster/logistic_regression_results.txt` - Results for cluster prediction
- `results/logistic_regression_model_cluster/logistic_regression_confusion_matrix.png` - Confusion matrix
- `results/logistic_regression_model_cluster/logistic_regression_feature_importance.png` - Feature importance
- `results/logistic_regression_model_cluster/feature_coefficients.tsv` - Feature coefficients

### Random Forest on Clusters:
- `results/random_forest_model_cluster/random_forest_results.txt` - Results for cluster prediction
- `results/random_forest_model_cluster/random_forest_confusion_matrix.png` - Confusion matrix
- `results/random_forest_model_cluster/random_forest_feature_importance.png` - Feature importance
- `results/random_forest_model_cluster/feature_importance.tsv` - Feature importances

### SVM on Clusters:
- `results/svm_cluster/svm_results.txt` - Results for cluster prediction
- `results/svm_cluster/svm_roc_curves.png` - ROC curves
- `results/svm_cluster/svm_confusion_matrix.png` - Confusion matrix

**Note:** Each cluster method (kmeans, GMM, spectral) was trained separately. The results above are for the most recent cluster method run.

---

## 4. Sample-Specific AUC Analysis

**Instruction:** Calculate sample-specific area under the ROC curve (AUC) across the predictive models. Generate a matrix of samples by models, where each cell is what class label the given model assigned to the given sample.

### Main Results:
- `results/sample_specific_auc/binary_predictions_matrix.tsv` - **Samples × Models matrix for binary predictions**
  - Each row is a sample, each column is a model
  - Values are predicted class labels (Trem2KO or WT)

- `results/sample_specific_auc/binary_sample_metrics.tsv` - **Per-sample metrics**
  - For each sample:
    - How many models predict each class label
    - Prediction stability (entropy, agreement)
    - Most common prediction and count

### Cluster Prediction Results:
- `results/sample_specific_auc/kmeans/kmeans_predictions_matrix.tsv` - Predictions matrix for kmeans clusters
- `results/sample_specific_auc/kmeans/kmeans_sample_metrics.tsv` - Per-sample metrics for kmeans
- `results/sample_specific_auc/gmm/gmm_predictions_matrix.tsv` - Predictions matrix for GMM clusters
- `results/sample_specific_auc/gmm/gmm_sample_metrics.tsv` - Per-sample metrics for GMM
- `results/sample_specific_auc/spectral/spectral_predictions_matrix.tsv` - Predictions matrix for spectral clusters
- `results/sample_specific_auc/spectral/spectral_sample_metrics.tsv` - Per-sample metrics for spectral

### Stability Correlation Analysis:
- `results/sample_specific_auc/stability_correlations.tsv` - **Statistical test results**
  - Correlation between cluster prediction stability and class label stability
  - Includes p-values with multiple test correction (Benjamini-Hochberg)

- `results/sample_specific_auc/sample_specific_analysis.png` - **Visualizations**
  - Agreement distribution
  - Entropy/stability distribution
  - Correlation plots
  - Model agreement distribution

---

## 5. Variable Gene Number Analysis

**Instruction:** Retrain each predictive model using different numbers of genes. Try 10, 100, 1000, and 10000 genes.

**Note:** This analysis may need to be run separately. If it exists, look for:

- `results/variable_genes_analysis/auc_by_gene_number.tsv` - **AUC for each model/gene combination**
  - Columns: model, n_genes, auc
  - Shows how AUC changes with number of genes

- `results/variable_genes_analysis/auc_vs_gene_number.png` - **Plot showing AUC vs number of genes**
  - One line per model
  - Shows performance trends

**To run this analysis:**
```bash
cd Assignment_4/scripts
python3 variable_gene_analysis.py --label-type binary
```

---

## 6. Heatmaps and Dendrograms

**Instruction:** Create a heatmap of the genes identified by the predictive methods, with annotation sidebar showing sample groups from Assignment 1. Include row and column dendrograms.

**Result Files:**
- `results/visualization/signature_heatmap.png` - **Main signature heatmap**
  - Shows only genes identified by predictive models (signature genes)
  - Annotation sidebar with Trem2KO vs WT groups
  - Row and column dendrograms
  - Legend and axis labels

- `results/visualization/signature_genes.txt` - **List of signature genes**
  - All genes identified as important by any model
  - Used in the signature heatmap

- `results/visualization/heatmap.png` - Basic heatmap (all 5000 genes, if created)
- `results/visualization/dendrogram.png` - Basic dendrogram (if created)

---

## Summary Table

| Instruction | Key Result Files |
|------------|------------------|
| **Subset to 5000 genes** | `expression_data_top5000.tsv` |
| **Predict Assignment 1 groups** | `results/*_model/*_results.txt`, `*_roc_curve.png`, `*_confusion_matrix.png` |
| **Predict Assignment 3 clusters** | `results/*_model_cluster/*_results.txt`, `*_confusion_matrix.png` |
| **Sample-specific AUC matrix** | `results/sample_specific_auc/binary_predictions_matrix.tsv` |
| **Per-sample metrics** | `results/sample_specific_auc/binary_sample_metrics.tsv` |
| **Stability correlation** | `results/sample_specific_auc/stability_correlations.tsv` |
| **Variable gene analysis** | `results/variable_genes_analysis/auc_by_gene_number.tsv` |
| **Signature heatmap** | `results/visualization/signature_heatmap.png` |

---

## Quick Reference: Which File Answers Which Question?

### "How many models predict each class label, for that sample?"
→ `results/sample_specific_auc/binary_sample_metrics.tsv`
- Columns: `n_models_predict_Trem2KO`, `n_models_predict_WT`

### "How many models predict the same cluster, for that sample?"
→ `results/sample_specific_auc/{kmeans|gmm|spectral}/{cluster}_sample_metrics.tsv`
- Column: `most_common_count` (how many models agree)

### "Does stability correlate between cluster and class predictions?"
→ `results/sample_specific_auc/stability_correlations.tsv`
- Shows correlation coefficients and p-values (with multiple test correction)

### "How did number of genes affect results?"
→ `results/variable_genes_analysis/auc_by_gene_number.tsv`
- Compare AUC across different gene numbers (10, 100, 1000, 10000)

### "What is the model performance (AUC) for each version?"
→ `results/variable_genes_analysis/auc_by_gene_number.tsv`
- Shows AUC for each model at each gene number

### "Does AUC increase or decrease with gene number?"
→ `results/variable_genes_analysis/auc_vs_gene_number.png`
- Visual plot showing trends

---

## Model Methods Implemented

Currently implemented:
1. ✅ **Logistic Regression** - `results/logistic_regression_model/`
2. ✅ **Random Forest** - `results/random_forest_model/`
3. ✅ **Support Vector Machine** - `results/svm/`

Still needed (if team has 4+ members):
- K Nearest Neighbors
- Naïve Bayes

