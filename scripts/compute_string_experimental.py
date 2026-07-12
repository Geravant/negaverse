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

    PYTHONPATH=. python scripts/compute_string_experimental.py --pairs-file pairs.tsv

`pairs.tsv` is `accession_a<TAB>accession_b` (UniProt), one pair per line.

Writes (merge, not clobber — same convention as the other annotation scripts):
    local-docs/annotations/string_experimental.tsv   node_a<TAB>node_b<TAB>score
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.string_channel import compute as _compute, write_merged, read_pairs_file

OUT = Path("local-docs/annotations/string_experimental.tsv")
CHANNEL = "experimental"


def compute(pairs: list[tuple[str, str]]) -> dict[frozenset[str], float]:
    return _compute(pairs, CHANNEL)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_string_experimental")
    ap.add_argument("--pairs-file", required=True, help="accession_a<TAB>accession_b, one pair per line")
    args = ap.parse_args(argv)

    pairs = read_pairs_file(args.pairs_file)
    print(f"{len(pairs)} pairs")

    scores = compute(pairs)
    total = write_merged(OUT, scores)
    print(f"wrote {len(scores)} new / {total} total pair scores to {OUT}")


if __name__ == "__main__":
    main()
