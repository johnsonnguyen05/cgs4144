import pandas as pd
import mygene

df = pd.read_csv("SRP119064/SRP119064.tsv", sep="\t")

# skip header
ensembl_ids = df.iloc[1:, 0].str.split(".").str[0].tolist()

# ensembl id -> gene symbol
mg = mygene.MyGeneInfo()
unique_ids = list(set(ensembl_ids))
results = mg.querymany(
    unique_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="mouse"
)

# Build mapping dict
mapping = {}
for r in results:
    if r.get("notfound"):
        continue
    symbol = r.get("symbol")
    if isinstance(symbol, list):
        symbol = symbol[0]  # take first if multiple hits
    mapping[r["query"]] = symbol

# Map IDs → symbols
mapped_ids = df.iloc[:, 0].iloc[1:].map(mapping)
df_filtered = df.iloc[1:][mapped_ids.notna()].copy()
df_filtered.iloc[:, 0] = mapped_ids[mapped_ids.notna()]

# Collapse duplicates by median
df_grouped = (
    df_filtered
    .groupby(df_filtered.columns[0], as_index=False)  # group by gene symbol
    .median(numeric_only=True)                       # median expression values
)

df_grouped.to_csv("expression_data.tsv", sep="\t", index=False)

print(f"Converted {len(mapping)} IDs. Output saved to expression_data.tsv")
