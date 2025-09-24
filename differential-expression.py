import pandas as pd

# Define file paths (replace with your actual file names)
data_file = "refinebio/SRP119064/SRP119064.tsv"
metadata_file = "refinebio/SRP119064/metadata_SRP119064.tsv"

# Read in the data and metadata
expression_df = pd.read_csv(data_file, sep='\t', index_col=0)
metadata_df = pd.read_csv(metadata_file, sep='\t', index_col=0) # Assuming sample IDs are the first column

# IMPORTANT: Ensure columns in expression_df match the index of metadata_df
# This step reorders the expression data to match the metadata order.
expression_df = expression_df[metadata_df.index]

# Check that the order is correct
# The following line should return True
all(expression_df.columns == metadata_df.index)