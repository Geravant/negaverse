# Estimates per-gene branch lengths on a FIXED master-tree topology, using
# RERconverge's own recommended method for exactly this scale of analysis
# (confirmed against vignettes/PhangornTreeBuildingWalkthrough.Rmd in
# github.com/nclark-lab/RERconverge): phangorn's pml/optim.pml maximum-
# likelihood branch-length fitting, wrapped by RERconverge's
# estimatePhangornTreeAll(). Writes ALL genes' trees into one combined
# GeneID<TAB>Newick file — the exact format scripts/rerconverge_runner.R's
# readTrees() call expects — so no separate Python-side concatenation step
# is needed.
#
# Fixing the topology (rather than searching for it per gene, e.g. via
# FastTree) avoids RERconverge discarding genes for "discordant tree
# topology": every gene tree here shares the master tree's topology by
# construction and differs only in relative branch lengths, which is exactly
# the signal Evolutionary Rate Covariation needs.
#
# The master tree (scripts/data/vertebrate_master_tree.nwk) topology reflects
# standard, uncontroversial vertebrate systematics for this project's 16-taxon
# species panel (see scripts/fetch_orthologs.py::SPECIES_PANEL); its branch
# lengths are an approximate divergence-time-proportional starting point, not
# a precise citation — optim.pml re-estimates each gene's actual branch
# lengths from its own alignment via maximum likelihood, so the master tree's
# own lengths only need to be a reasonable optimizer seed.
#
# Usage:
#   Rscript scripts/estimate_phangorn_trees.R <alignment_dir> <master_tree> <out_trees_file>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript estimate_phangorn_trees.R <alignment_dir> <master_tree> <out_trees_file>")
}
alignment_dir <- args[1]
master_tree <- args[2]
out_trees_file <- args[3]

suppressPackageStartupMessages(library(RERconverge))

estimatePhangornTreeAll(alndir = alignment_dir, treefile = master_tree,
                        output.file = out_trees_file,
                        format = "fasta", type = "DNA", submodel = "GTR")

cat(sprintf("wrote gene trees to %s\n", out_trees_file))
