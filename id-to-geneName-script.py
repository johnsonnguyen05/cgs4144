import pandas as pd
import mygene

# Load expression matrix
df = pd.read_csv("refinebio/SRP119064/SRP119064.tsv", sep="\t")

# Strip Ensembl version numbers from first column
df.iloc[:,0] = df.iloc[:,0].str.split(".").str[0]
ensembl_ids = df.iloc[:,0].tolist()

# Query MyGene.info (mouse Ensembl IDs → gene symbols)
mg = mygene.MyGeneInfo()
unique_ids = list(set(ensembl_ids))
results = mg.querymany(
    unique_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="mouse"
)

# Build mapping dict (handle duplicates and missing IDs)
mapping = {}
for r in results:
    if r.get("notfound"):
        continue
    symbol = r.get("symbol")
    if isinstance(symbol, list):
        symbol = symbol[0]  # take the first if multiple
    mapping[r["query"]] = symbol

# Map IDs → symbols, keep Ensembl ID if not found
df.iloc[:,0] = df.iloc[:,0].map(mapping).fillna(df.iloc[:,0])

# Save to new file
df.to_csv("expression_with_gene_names.tsv", sep="\t", index=False)

print(f"Converted {len(mapping)} IDs. Output saved to expression_with_gene_names.tsv")
