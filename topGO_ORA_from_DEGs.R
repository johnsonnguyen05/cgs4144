#!/usr/bin/env Rscript
## topGO over-representation analysis (ORA) from DE results
## Reads: results/SRP119064_diff_expr_results.tsv
## Writes: results/topGO_ORA_upregulated_BP.tsv and top 15 table

required_pkgs <- c('topGO', 'org.Mm.eg.db', 'AnnotationDbi')
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly=TRUE)]
if (length(missing_pkgs) > 0) {
  stop(paste0('Missing R/Bioconductor packages: ', paste(missing_pkgs, collapse=', '),
              '\nInstall with:\nif (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")\n',
              'BiocManager::install(c("', paste(missing_pkgs, collapse='","'), '"), ask=FALSE)'))
}

library(topGO)
library(org.Mm.eg.db)
library(AnnotationDbi)

de_file <- file.path('results','SRP119064_diff_expr_results.tsv')
if (!file.exists(de_file)) stop('DE results not found at results/SRP119064_diff_expr_results.tsv. Run differential-expression pipeline first.')

de <- read.delim(de_file, sep='\t', stringsAsFactors=FALSE, check.names=FALSE)
if (!all(c('Gene','log2FoldChange','padj') %in% colnames(de))) stop('Expected columns Gene, log2FoldChange, padj in DE results')

# remove rows with NA in key columns
de <- de[!is.na(de$log2FoldChange) & !is.na(de$padj), ]

# select upregulated genes: padj < 0.05 and log2FC >= 1
padj_cut <- 0.05
lfc_cut <- 1.0
up_genes <- de$Gene[de$padj < padj_cut & de$log2FoldChange >= lfc_cut]
cat('Selected', length(up_genes), 'upregulated genes using padj <', padj_cut, 'and log2FC >=', lfc_cut, '\n')
if (length(up_genes) == 0) {
  warning('No upregulated genes found with the chosen thresholds; relaxing to padj < 0.05 & log2FC > 0')
  up_genes <- de$Gene[de$padj < padj_cut & de$log2FoldChange > 0]
  cat('Now selected', length(up_genes), 'upregulated genes (padj<', padj_cut, ' & log2FC>0)\n')
  if (length(up_genes) == 0) stop('Still no upregulated genes found; cannot run ORA')
}

# universe = all genes tested in DE
universe_genes <- de$Gene
cat('Universe size (genes tested):', length(universe_genes), '\n')

# Create a named factor allGenes required by topGO (1 for selected genes, 0 otherwise)
allGenes <- factor(as.integer(universe_genes %in% up_genes))
names(allGenes) <- universe_genes

# Build topGOdata object: default ontology = BP (change if desired)
ontology <- 'BP'
GOdata <- new('topGOdata', ontology = ontology,
              allGenes = allGenes,
              geneSel = function(x) x == 1,
              annot = annFUN.org, mapping = 'org.Mm.eg.db', ID = 'SYMBOL', nodeSize = 10)

# Run tests: classic Fisher and elim Fisher
resultClassic <- runTest(GOdata, algorithm='classic', statistic='fisher')
resultElim <- runTest(GOdata, algorithm='elim', statistic='fisher')

## Get results table (topNodes large to retrieve many terms)
res_tab <- GenTable(GOdata, classicFisher = resultClassic, elimFisher = resultElim,
                   orderBy='classicFisher', ranksOf = 'classicFisher', topNodes = 200)

# If GenTable returned no rows, write empty outputs and exit gracefully
if (is.null(res_tab) || nrow(res_tab) == 0) {
  warning('topGO returned 0 GO terms for the selected gene set. Writing empty result files.')
  if (!dir.exists('results')) dir.create('results')
  empty_df <- data.frame(Term=character(0), GO.ID=character(0), Ontology=character(0), Annotated=integer(0), Significant=integer(0), Expected=numeric(0), classicFisher_pval=numeric(0), classicFisher_adj_pval=numeric(0), negLog10_adjClassic=numeric(0), elimFisher_pval=numeric(0), elimFisher_adj_pval=numeric(0), negLog10_adjElim=numeric(0), stringsAsFactors=FALSE)
  write.table(empty_df, file=file.path('results', paste0('topGO_ORA_upregulated_', ontology, '.tsv')), sep='\t', quote=FALSE, row.names=FALSE)
  write.table(empty_df, file=file.path('results', paste0('topGO_ORA_upregulated_', ontology, '_top15.tsv')), sep='\t', quote=FALSE, row.names=FALSE)
  stop('No GO terms returned by topGO; exiting.')
}

# GenTable returns p-values as character strings (may include '<'); convert to numeric
clean_pval <- function(x) {
  x2 <- gsub('<', '', x)
  x2 <- as.numeric(x2)
  return(x2)
}

# Detect which columns GenTable returned for term and GO ID
res_names <- tolower(colnames(res_tab))
term_col <- NULL
goid_col <- NULL
if ('term' %in% res_names) term_col <- colnames(res_tab)[which(res_names == 'term')[1]]
if ('go.term' %in% res_names) term_col <- colnames(res_tab)[which(res_names == 'go.term')[1]]
if ('go.term' %in% colnames(res_tab)) term_col <- 'GO.term'
if (is.null(term_col)) {
  # fallback to first column that is character and not GO.ID
  char_cols <- sapply(res_tab, is.character)
  possible <- names(res_tab)[char_cols]
  term_col <- possible[1]
}
if ('go.id' %in% res_names) goid_col <- colnames(res_tab)[which(res_names == 'go.id')[1]]
if (is.null(goid_col) && any(grepl('go.id', colnames(res_tab), ignore.case=TRUE))) goid_col <- colnames(res_tab)[grepl('go.id', colnames(res_tab), ignore.case=TRUE)][1]
if (is.null(goid_col)) {
  # try common name
  if ('go.id' %in% colnames(res_tab)) goid_col <- 'GO.ID'
}

# If specific fisher columns exist, convert; otherwise try to find them
if ('classicFisher' %in% colnames(res_tab)) {
  res_tab$classicFisher_pval <- clean_pval(res_tab$classicFisher)
} else if ('classic' %in% colnames(res_tab)) {
  res_tab$classicFisher_pval <- clean_pval(res_tab$classic)
} else {
  res_tab$classicFisher_pval <- rep(NA, nrow(res_tab))
}
if ('elimFisher' %in% colnames(res_tab)) {
  res_tab$elimFisher_pval <- clean_pval(res_tab$elimFisher)
} else if ('elim' %in% colnames(res_tab)) {
  res_tab$elimFisher_pval <- clean_pval(res_tab$elim)
} else {
  res_tab$elimFisher_pval <- rep(NA, nrow(res_tab))
}

# Adjust p-values (BH) for the classic Fisher p-values
res_tab$classicFisher_adj_pval <- p.adjust(res_tab$classicFisher_pval, method='BH')
res_tab$elimFisher_adj_pval <- p.adjust(res_tab$elimFisher_pval, method='BH')

# compute expected count = Annotated * (k / N), where k = number of selected genes, N = universe
k_selected <- length(up_genes)
N_universe <- length(universe_genes)
res_tab$Annotated <- as.integer(res_tab$Annotated)
res_tab$Significant <- as.integer(res_tab$Significant)
res_tab$Expected <- res_tab$Annotated * (k_selected / N_universe)

# Add -log10(adj p-value) for display
res_tab$negLog10_adjClassic <- -log10(res_tab$classicFisher_adj_pval + 1e-300)
res_tab$negLog10_adjElim <- -log10(res_tab$elimFisher_adj_pval + 1e-300)

## Diagnostics: show what GenTable returned
cat('GenTable returned', nrow(res_tab), 'rows and columns:', paste(colnames(res_tab), collapse=', '), '\n')

# Helper to find a column name by several candidate patterns
find_col <- function(df, patterns) {
  cols <- colnames(df)
  cols_l <- tolower(cols)
  for (p in patterns) {
    i <- grep(p, cols_l)
    if (length(i) > 0) return(cols[i[1]])
  }
  return(NULL)
}

# Detect term and GO ID columns
term_col <- find_col(res_tab, c('go.term','term','description'))
goid_col <- find_col(res_tab, c('go.id','goid','go.id.','goid'))
annot_col <- find_col(res_tab, c('annotated','annot'))
signif_col <- find_col(res_tab, c('significant','signif'))

if (is.null(term_col)) {
  warning('Could not detect a GO term/description column in GenTable output; using first column as Term')
  term_col <- colnames(res_tab)[1]
}
if (is.null(goid_col)) {
  warning('Could not detect a GO.ID column in GenTable output; GO.ID will be empty')
}
if (is.null(annot_col) || is.null(signif_col)) {
  warning('Annotated or Significant columns not detected; attempting to use columns named Annotated/Significant if present')
  if ('Annotated' %in% colnames(res_tab)) annot_col <- 'Annotated'
  if ('Significant' %in% colnames(res_tab)) signif_col <- 'Significant'
}

# Extract columns safely
term_vec <- as.character(res_tab[[term_col]])
goid_vec <- if (!is.null(goid_col)) as.character(res_tab[[goid_col]]) else rep(NA, length(term_vec))
annot_vec <- if (!is.null(annot_col)) as.integer(res_tab[[annot_col]]) else rep(NA_integer_, length(term_vec))
signif_vec <- if (!is.null(signif_col)) as.integer(res_tab[[signif_col]]) else rep(NA_integer_, length(term_vec))

# Build output table using the processed p-values and computed fields
out_tab <- data.frame(
  Term = term_vec,
  GO.ID = goid_vec,
  Ontology = ontology,
  Annotated = annot_vec,
  Significant = signif_vec,
  Expected = round(res_tab$Expected,3),
  classicFisher_pval = signif(res_tab$classicFisher_pval, 3),
  classicFisher_adj_pval = signif(res_tab$classicFisher_adj_pval, 3),
  negLog10_adjClassic = signif(res_tab$negLog10_adjClassic, 3),
  elimFisher_pval = signif(res_tab$elimFisher_pval, 3),
  elimFisher_adj_pval = signif(res_tab$elimFisher_adj_pval, 3),
  negLog10_adjElim = signif(res_tab$negLog10_adjElim, 3),
  stringsAsFactors = FALSE
)

if (!dir.exists('results')) dir.create('results')
out_file <- file.path('results', paste0('topGO_ORA_upregulated_', ontology, '.tsv'))
write.table(out_tab, file = out_file, sep='\t', quote=FALSE, row.names=FALSE)
cat('Full ORA results saved to:', out_file, '\n')

# Also save top 15 terms (by adjusted classic p-value)
top15 <- out_tab[order(as.numeric(as.character(out_tab$classicFisher_adj_pval))), ]
top15 <- head(top15, 15)
top15_file <- file.path('results', paste0('topGO_ORA_upregulated_', ontology, '_top15.tsv'))
write.table(top15, file = top15_file, sep='\t', quote=FALSE, row.names=FALSE)
cat('Top 15 ORA terms saved to:', top15_file, '\n')

cat('Script finished. You can visualize top terms (negLog10_adjClassic) as a barplot similar to your provided figure.')
