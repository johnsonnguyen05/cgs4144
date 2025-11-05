#!/usr/bin/env python3
"""Subset expression_data.tsv to the top N most variable genes.

Writes output to expression_data_top5000.tsv in the Assignment 3 directory.

Usage: python subset_top_variable_genes.py [--input PATH] [--output PATH] [--n N]
"""
import argparse
import os
import sys
import pandas as pd


def main(argv):
    p = argparse.ArgumentParser(description="Subset expression data to most variable genes")
    p.add_argument("--input", "-i", default="expression_data.tsv", help="Input expression TSV (genes x samples) [default: expression_data.tsv]")
    p.add_argument("--output", "-o", default="Assignment_3/expression_data_top5000.tsv", help="Output TSV path")
    p.add_argument("--n", "-n", type=int, default=5000, help="Number of top variable genes to keep")
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(2)

    print(f"Reading input: {args.input}")
    df = pd.read_csv(args.input, sep="\t", header=0)

    # Heuristic: first column is gene identifier, remaining are samples
    if df.shape[1] < 2:
        print("Input appears to have fewer than 2 columns. Expecting gene ids + sample columns.")
        sys.exit(3)

    gene_col = df.columns[0]
    expr = df.iloc[:, 1:]

    # Convert expression data to numeric (coerce non-numeric to NaN)
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # Compute variance across samples for each gene (skip NaNs)
    variances = expr.var(axis=1, skipna=True)

    # Rank and select top N
    n_keep = min(args.n, len(variances))
    top_idx = variances.nlargest(n_keep).index

    df_top = df.loc[top_idx]

    # Write output, preserving gene id column and sample columns
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_top.to_csv(args.output, sep="\t", index=False)

    print(f"Wrote {len(df_top)} genes to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
