# cgs4144

**Team Members:** Richard Hou, Johnson Nguyen, Rayyan Shaikh

**Data:** [Loss of Trem2 in microglia leads to widespread disruption of cell coexpression networks in mouse brain](https://pubmed.ncbi.nlm.nih.gov/29906661/)

**Question:** Does loss of Trem2 disrupt brain cell communication networks in a way that increases vulnerability to Alzheimer’s disease?

Pip packages installed: pandas and mygene
Ran script _id-to-geneName-script.py_ in order to replace ensemble id with gene names, output results to expression_with_gene_names.tsv

To analyze expression matrix, install matplotlib and scipy pip package

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

pip install seaborn, umap, umap-learn, scikit-learn
