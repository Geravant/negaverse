# Thin RERconverge wrapper, invoked as a subprocess from
# scripts/compute_evolutionary_coupling.py (same "shell out to the real tool"
# pattern as compute_pocket_descriptors.py calling fpocket).
#
# Reads a combined gene-trees file (GeneID<TAB>Newick, one gene per line — the
# exact format RERconverge's own readTrees() expects, confirmed against
# inst/extdata/subsetMammalGeneTrees.txt in github.com/nclark-lab/RERconverge),
# estimates a master tree + per-branch relative evolutionary rates (RER) for
# every gene via readTrees()/getAllResiduals() (verified against
# vignettes/FullWalkthroughUTD.Rmd in that repo — those two calls are all we
# need; the phenotype-correlation functions like correlateWithBinaryPhenotype
# don't apply here, since we're correlating two genes' RER vectors against
# each other, not against a trait), and writes the resulting gene x branch RER
# matrix to CSV for the Python side to read and correlate pairwise.
#
# Usage:
#   Rscript scripts/rerconverge_runner.R <trees_file> <out_csv> [min_trees_all] [min_valid]
#
# min_trees_all (default 20, RERconverge's own default): the minimum number of
# full-species-coverage gene trees needed before readTrees will estimate its
# own master tree.
#
# min_valid (default 20, RERconverge's own default for getAllResiduals' own
# min.valid): the minimum number of non-NA (gene, branch) values a branch
# needs before its weighted-regression correction is computed at all — below
# that, getAllResiduals' internal diagnostic plot gets empty data and errors
# out on a bad plot.window() call (confirmed empirically: happened with 19
# genes total, disappeared once genes >> min.valid). Real calibration runs
# (hundreds of unique proteins) should clear the default comfortably; lower
# both for small smoke tests.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript rerconverge_runner.R <trees_file> <out_csv> [min_trees_all] [min_valid]")
}
trees_file <- args[1]
out_csv <- args[2]
min_trees_all <- if (length(args) >= 3) as.integer(args[3]) else 20L
min_valid <- if (length(args) >= 4) as.integer(args[4]) else 20L

suppressPackageStartupMessages(library(RERconverge))

# getAllResiduals() plots a diagnostic boxplot as a side effect; we don't use
# it (the Python side generates its own ROC/histogram plots) and don't want a
# stray Rplots.pdf appearing wherever this script happens to be invoked from.
# pdf(file = NULL) opens R's real null graphics device — plot calls succeed
# and are simply discarded, rather than either erroring or leaving a file.
pdf(file = NULL)

trees <- readTrees(trees_file, minTreesAll = min_trees_all)
rer <- getAllResiduals(trees, useSpecies = trees$masterTree$tip.label,
                        transform = "sqrt", n.pcs = 0, use.weights = TRUE,
                        weights = NULL, norm = "scale", min.valid = min_valid)

write.csv(rer, file = out_csv)
cat(sprintf("wrote RER matrix (%d genes x %d branches) to %s\n",
            nrow(rer), ncol(rer), out_csv))
