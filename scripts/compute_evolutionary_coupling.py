"""Compute Evolutionary Rate Covariation (ERC) between specific protein pairs,
for the evolutionary_coupling_absence rule's `a.evolutionary_coupling_score_with_b`
field.

Orchestrates the full pipeline: scripts/fetch_orthologs.py (Ensembl orthologs,
UniProt-keyed) -> scripts/build_gene_alignments.py (MAFFT per gene) ->
scripts/estimate_phangorn_trees.R (RERconverge's fixed-master-topology
branch-length estimation via phangorn — see that script's docstring for why
this replaced an earlier free-topology-per-gene approach) ->
scripts/rerconverge_runner.R (RERconverge's readTrees + getAllResiduals,
subprocess-invoked the same way compute_pocket_descriptors.py shells out to
fpocket) -> Pearson correlation between the two genes' relative-evolutionary-
rate (RER) vectors, restricted to branches where both have a value.

**Quality gate**, mirroring fpocket's "no confident structure -> no score" and
EVcomplex's Neff/L cutoff: a pair's ERC score is only computed/written if both
genes have RER values for at least MIN_SHARED_BRANCHES branches — a
correlation over fewer points is unstable noise, not signal, so the pair's
field is left absent (abstain) rather than populated with a low-confidence
number.

**Scope note**: the ortholog panel and master tree
(scripts/data/vertebrate_master_tree.nwk) are vertebrate-specific, matching
this project's human PPI calibration benchmarks (DRYAD, UPNA-PPI). Bacterial
or viral query proteins simply won't have an Ensembl vertebrate gene mapping
at all, so they abstain gracefully (no score, no fabrication) rather than
producing a meaningless result — this pipeline does not attempt evolutionary
coupling for non-vertebrate proteins; that would need its own reference
panel and phylogeny (and, for viruses, is likely not a well-posed question at
all given how fast viral proteins evolve and how shallow their cross-species
orthology is).

    PYTHONPATH=. python scripts/compute_evolutionary_coupling.py --pairs-file pairs.tsv

`pairs.tsv` is `accession_a<TAB>accession_b` (UniProt), one pair per line.

Writes (merge, not clobber — same convention as the hydrophobicity/pocket
scripts):
    local-docs/annotations/evolutionary_coupling.tsv   node_a<TAB>node_b<TAB>score
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from scripts.build_gene_alignments import build_many
from scripts.fetch_orthologs import map_accessions_to_genes, fetch_many_genes

OUT = Path("local-docs/annotations/evolutionary_coupling.tsv")
GENE_TREES_DIR = Path("local-docs/gene_trees")
MASTER_TREE = Path("scripts/data/vertebrate_master_tree.nwk")
TREE_ESTIMATOR = Path("scripts/estimate_phangorn_trees.R")
RUNNER = Path("scripts/rerconverge_runner.R")
MIN_SHARED_BRANCHES = 10   # below this, a Pearson correlation is noise, not signal


def _read_trees_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip() and "\t" in line:
            gid, newick = line.split("\t", 1)
            out[gid] = newick
    return out


def _estimate_trees(alignment_dir: Path, out_trees_file: Path) -> int:
    """Runs estimate_phangorn_trees.R **only for genes not already in
    out_trees_file** — phangorn's ML branch-length fitting is the slow step
    (~5-15s/gene here), so re-estimating already-done genes on every call
    (e.g. once per dataset in the calibration script) wastes real time as the
    alignment directory grows across calls. Merges new trees into the
    existing file rather than overwriting, same convention as every other
    annotation script's _write_merged. Returns the total gene count (existing
    + newly estimated) after this call."""
    existing = _read_trees_file(out_trees_file)
    all_genes = {p.stem for p in alignment_dir.glob("*.fasta")}
    new_genes = sorted(all_genes - set(existing))
    if not new_genes:
        print(f"  {len(existing)}/{len(existing)} genes already have a tree (cached); nothing new to estimate")
        return len(existing)
    print(f"  {len(existing)} genes already have a cached tree; "
          f"estimating {len(new_genes)} new gene(s)")

    # estimatePhangornTreeAll scans its whole alndir — build a scratch dir of
    # symlinks to just the new alignments so it doesn't reprocess cached ones.
    scratch_dir = alignment_dir.parent / f"{alignment_dir.name}_new_scratch"
    if scratch_dir.exists():
        for f in scratch_dir.glob("*"):
            f.unlink()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for gene in new_genes:
        (scratch_dir / f"{gene}.fasta").symlink_to((alignment_dir / f"{gene}.fasta").resolve())

    new_trees_file = out_trees_file.with_name(out_trees_file.stem + "_new.tsv")
    r = subprocess.run(
        ["Rscript", str(TREE_ESTIMATOR), str(scratch_dir), str(MASTER_TREE), str(new_trees_file)],
        capture_output=True, text=True)
    if r.returncode != 0 or not new_trees_file.exists():
        print(f"  estimate_phangorn_trees.R failed:\n{r.stderr[-2000:]}")
        return len(existing)

    new_trees = _read_trees_file(new_trees_file)
    merged = {**existing, **new_trees}
    with open(out_trees_file, "w") as fh:
        for gid in sorted(merged):
            fh.write(f"{gid}\t{merged[gid]}\n")
    return len(merged)


def _run_rerconverge(trees_file: Path, out_csv: Path, min_trees_all: int,
                      min_valid: int) -> dict[str, dict[str, float]]:
    """Runs rerconverge_runner.R, returns {gene_id: {branch: rer_value}} parsed
    from the RER-matrix CSV it writes (genes as rows, branches as columns)."""
    r = subprocess.run(
        ["Rscript", str(RUNNER), str(trees_file), str(out_csv),
         str(min_trees_all), str(min_valid)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  rerconverge_runner.R failed:\n{r.stderr}")
        return {}

    rer: dict[str, dict[str, float]] = {}
    with open(out_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        branches = header[1:]
        for row in reader:
            gene_id, vals = row[0], row[1:]
            rer[gene_id] = {}
            for branch, val in zip(branches, vals):
                if val not in ("", "NA"):
                    try:
                        rer[gene_id][branch] = float(val)
                    except ValueError:
                        continue
    return rer


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom else 0.0


def _pair_correlation(rer_a: dict[str, float], rer_b: dict[str, float]) -> tuple[float | None, int]:
    shared = sorted(set(rer_a) & set(rer_b))
    if len(shared) < MIN_SHARED_BRANCHES:
        return None, len(shared)
    x = [rer_a[b] for b in shared]
    y = [rer_b[b] for b in shared]
    return round(_pearson(x, y), 4), len(shared)


def compute(pairs: list[tuple[str, str]], min_trees_all: int = 20,
            min_valid: int = 20) -> dict[frozenset[str], float]:
    """Returns {frozenset({acc_a, acc_b}): erc_score} for pairs that pass the
    MIN_SHARED_BRANCHES quality gate. Pairs whose proteins lack a usable
    Ensembl mapping, ortholog set, gene tree, or RER estimate are simply
    absent — no fabrication, matching every other annotation script here."""
    accessions = sorted({acc for pair in pairs for acc in pair})
    acc_to_gene = map_accessions_to_genes(accessions)

    gene_ids = sorted(set(acc_to_gene.values()))
    print(f"fetching orthologs for {len(gene_ids)} genes...")
    fetch_many_genes(gene_ids)          # populates local-docs/orthologs/*.json (cached)

    print(f"aligning {len(gene_ids)} genes...")
    alignment_paths = build_many(gene_ids)
    print(f"  {len(alignment_paths)}/{len(gene_ids)} genes produced a usable alignment")
    if not alignment_paths:
        print("no alignments to build trees from")
        return {}
    alignment_dir = next(iter(alignment_paths.values())).parent

    trees_file = GENE_TREES_DIR / "_combined_trees.tsv"
    out_csv = GENE_TREES_DIR / "_rer_matrix.csv"
    GENE_TREES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"estimating branch lengths on the fixed master topology for "
          f"{len(alignment_paths)} genes...")
    n_written = _estimate_trees(alignment_dir, trees_file)
    if n_written == 0:
        print("no gene trees produced")
        return {}
    print(f"running RERconverge on {n_written} gene trees...")
    rer = _run_rerconverge(trees_file, out_csv, min_trees_all, min_valid)
    print(f"  RER estimated for {len(rer)}/{n_written} genes")

    scores: dict[frozenset[str], float] = {}
    n_gated = 0
    for acc_a, acc_b in pairs:
        gene_a, gene_b = acc_to_gene.get(acc_a), acc_to_gene.get(acc_b)
        if not gene_a or not gene_b or gene_a not in rer or gene_b not in rer:
            continue
        score, n_shared = _pair_correlation(rer[gene_a], rer[gene_b])
        if score is None:
            n_gated += 1
            continue
        scores[frozenset((acc_a, acc_b))] = score
    print(f"scored {len(scores)}/{len(pairs)} pairs "
          f"({n_gated} dropped by the <{MIN_SHARED_BRANCHES}-shared-branch quality gate)")
    return scores


def _write_merged(path: Path, new_values: dict[frozenset[str], float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[frozenset[str], float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                a, b, v = line.split("\t")[:3]
                merged[frozenset((a, b))] = float(v)
    merged.update(new_values)
    with open(path, "w") as fh:
        for pair in sorted(merged, key=lambda p: sorted(p)):
            a, b = sorted(pair)
            fh.write(f"{a}\t{b}\t{merged[pair]}\n")
    return len(merged)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_evolutionary_coupling")
    ap.add_argument("--pairs-file", required=True, help="accession_a<TAB>accession_b, one pair per line")
    ap.add_argument("--min-trees-all", type=int, default=20,
                     help="RERconverge's own master-tree-estimation minimum; lower for small smoke tests")
    ap.add_argument("--min-valid", type=int, default=20,
                     help="RERconverge's own min.valid (getAllResiduals); lower for small smoke tests")
    args = ap.parse_args(argv)

    pairs = []
    for line in Path(args.pairs_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            a, b = line.split("\t")[:2]
            pairs.append((a, b))
    print(f"{len(pairs)} pairs")

    scores = compute(pairs, min_trees_all=args.min_trees_all, min_valid=args.min_valid)
    total = _write_merged(OUT, scores)
    print(f"wrote {len(scores)} new / {total} total pair scores to {OUT}")


if __name__ == "__main__":
    main()
