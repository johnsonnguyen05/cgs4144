# cgs4144

**Team Members:** Richard Hou, Johnson Nguyen, Rayyan Shaikh

**Data:** [Loss of Trem2 in microglia leads to widespread disruption of cell coexpression networks in mouse brain](https://pubmed.ncbi.nlm.nih.gov/29906661/)

**Question:** Does loss of Trem2 disrupt brain cell communication networks in a way that increases vulnerability to Alzheimer’s disease?

installed packges from requirements.txt
provide SRP119064 folder in root

run id-to-geneName-script.py to get new tsv (expression_with_gene_names.tsv)

run expression-matrix-script.py to get summary of findings

summary of findings:
Matrix size: 41249 genes × 483 samples
Summary of per-gene median expression (log2 scale):
count 41249.000000
mean 1.308183
std 1.315714
min 0.195652
25% 0.196322
50% 0.920552
75% 2.144825
max 7.519327
dtype: float64

run generate-plots.py to get plot
