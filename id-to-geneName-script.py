import pandas as pd
import mygene

# Load expression matrix
df = pd.read_csv("refinebio/SRP119064/SRP119064.tsv", sep="\t")

# Skip the first row of the first column when processing Ensembl IDs
ensembl_ids = df.iloc[1:, 0].str.split(".").str[0].tolist()  # skip row 0

# Query MyGene.info (mouse Ensembl IDs → gene symbols)
mg = mygene.MyGeneInfo()
unique_ids = list(set(ensembl_ids))
results = mg.querymany(
    unique_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="mouse"
)

# Build mapping dict (handle duplicates)
mapping = {}
for r in results:
    if r.get("notfound"):
        continue
    symbol = r.get("symbol")
    if isinstance(symbol, list):
        symbol = symbol[0]  # take the first if multiple
    mapping[r["query"]] = symbol

# Map IDs → symbols
# Only keep rows where the mapping exists
mapped_ids = df.iloc[:,0].iloc[1:].map(mapping)  # skip first row
df_filtered = df.iloc[1:][mapped_ids.notna()].copy()  # drop unmapped rows
df_filtered.iloc[:,0] = mapped_ids[mapped_ids.notna()]  # replace with symbols

# Save to new file
df_filtered.to_csv("expression_with_gene_names.tsv", sep="\t", index=False)

print(f"Converted {len(mapping)} IDs. Output saved to expression_with_gene_names.tsv")
