"""Compute STRING's `experimental` channel score between specific protein
pairs, for `rules/ppi.yaml`'s `string_low_confidence_non_interaction` rule
(`experimental < 0.15` -> safer_negative): below STRING's own lowest
reporting cutoff for direct experimental evidence, a weaker/graded signal.

The opposite tail (`experimental > 0.9`, strong direct evidence a pair really
does interact) is handled by a *different* script,
scripts/build_known_positive_sources.py, as a known-positive source for
`KnownPositiveVeto` (`rules/sources.yaml`'s `string_experimental_high_confidence`)
rather than a rule here — see that script's docstring and `rules/SOURCES.md`
for why. This script only ever needs to answer "what's the score for these
specific requested pairs," while that one needs to scan the whole STRING
file once for every pair clearing a fixed threshold — different shapes of
problem, different scripts, sharing plumbing via scripts/string_channel.py.

Unlike `cooccurence` (phylogenetic profile, an indirect evolutionary-coupling
proxy — tested as a candidate PPI signal against DRYAD/UPNA-PPI and found no
reliable separation; removed once that was settled), `experimental` is
STRING's most direct binding-evidence channel (physical assays: yeast two-hybrid, affinity
capture, etc.), which is why it's the right channel here and not
`combined_score` (which blends in weaker, indirect evidence like
text-mining and coexpression that shouldn't count as strongly either way).

Uses the shared ID-mapping and links-file-streaming plumbing in
scripts/string_channel.py.

Two modes:

    PYTHONPATH=. python scripts/compute_string_experimental.py --pairs-file pairs.tsv
    PYTHONPATH=. python scripts/compute_string_experimental.py --ids-file huri_nodes.txt

`--pairs-file` scores specific, already-known `accession_a<TAB>accession_b`
pairs (UniProt or Ensembl gene ids), one pair per line — the original mode,
still here for ad hoc/small requests.

`--ids-file` instead takes one node id per line for a WHOLE graph (e.g. every
HuRI gene id) and scores every pair *within* that node set that STRING's
`experimental` channel has any row for — the bulk-precompute path. Candidate
pairs a pipeline run scores are a random sample of a graph's non-edges,
redrawn every run with no fixed pair list to hand this in advance; but every
candidate's endpoints are always nodes of that graph, so scanning by node
membership (not a pre-listed pair) covers whatever any run could ever draw,
in one pass, regardless of graph size or candidate-generation seed. Same
restrict-by-node-set pattern already used by
scripts/build_known_positive_sources.py::build_string for the known-positive
side of this same channel.

Writes (merge, not clobber — same convention as the other annotation
scripts, and safe to accumulate across multiple runs/graphs/input datasets:
UniProt accessions and Ensembl gene ids never collide, so a HuRI run's rows
and a DRYAD/SARS run's rows coexist in the same file without re-scanning
what's already there):
    local-docs/annotations/string_experimental.tsv   node_a<TAB>node_b<TAB>score
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.string_channel import (
    compute as _compute,
    compute_for_node_set as _compute_for_node_set,
    write_merged,
    read_pairs_file,
)

OUT = Path("local-docs/annotations/string_experimental.tsv")
CHANNEL = "experimental"


def compute(pairs: list[tuple[str, str]]) -> dict[frozenset[str], float]:
    return _compute(pairs, CHANNEL)


def compute_for_node_set(node_ids: list[str]) -> dict[frozenset[str], float]:
    return _compute_for_node_set(node_ids, CHANNEL)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_string_experimental")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs-file", help="accession_a<TAB>accession_b, one pair per line")
    group.add_argument("--ids-file", help="one node id per line; scores every in-set pair STRING reports")
    args = ap.parse_args(argv)

    if args.pairs_file:
        pairs = read_pairs_file(args.pairs_file)
        print(f"{len(pairs)} pairs")
        scores = compute(pairs)
    else:
        ids = sorted({l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()})
        print(f"{len(ids)} node ids")
        scores = compute_for_node_set(ids)

    total = write_merged(OUT, scores)
    print(f"wrote {len(scores)} new / {total} total pair scores to {OUT}")


if __name__ == "__main__":
    main()
