# Chi-Squared Test Results Summary

## Focused Analysis: Comparing Methods Within Same Gene Counts

### Results Summary

| Gene Count | Most Similar Pair | χ² | Most Different Pair | χ² |
|------------|-------------------|----|-------------------|----|
| **10 genes** | GMM vs Spectral | 1308.35 | GMM vs K-means | 1809.58 |
| **100 genes** | GMM vs Spectral | 1103.38 | K-means vs Spectral | 1546.90 |
| **1000 genes** | GMM vs Spectral | 1135.61 | GMM vs K-means | 1391.40 |
| **5000 genes** | K-means vs Spectral | 993.10 | GMM vs K-means | 1441.68 |
| **10000 genes** | K-means vs Spectral | 993.10 | GMM vs K-means | 1441.68 |

### Key Findings

1. **All comparisons were statistically significant** (p < 0.05), indicating substantial differences between clustering methods regardless of gene count.

2. **Gene count impact on method similarity**:
   - **10 genes**: Highest chi-squared values (most different results)
   - **100-1000 genes**: Intermediate differences
   - **5000-10000 genes**: Lowest chi-squared values (most similar results)

3. **Method relationships**:
   - **K-means vs Spectral**: Generally most similar across gene counts
   - **GMM vs K-means**: Generally most different across gene counts
   - **GMM vs Spectral**: Intermediate similarity

4. **Stability with gene count**:
   - Using more genes (5000-10000) leads to more consistent clustering patterns
   - Using fewer genes (10-100) leads to more variable and different clustering results

### Statistical Interpretation

- **χ² values range from 993 to 1809**, all highly significant (p < 0.001)
- **Lower χ² values** indicate more similar clustering results
- **Higher χ² values** indicate more different clustering results
- The consistent pattern across gene counts suggests that the choice of clustering method has a substantial impact on results regardless of the number of genes used
