"""Shared plumbing for reading a single named channel out of STRING's
per-species `protein.links.detailed` file for specific UniProt-accession
pairs. Used by scripts/compute_string_experimental.py (`experimental`
channel) — a `cooccurence`-channel counterpart existed here too, but was
removed after testing found no reliable separation on DRYAD/UPNA-PPI as a
candidate evolutionary-coupling signal. Kept as shared plumbing (not folded
into that one script) in case another STRING channel is worth testing later.

Both files are already downloaded, purely local lookups (no network calls):
  - local-docs/string/9606.protein.aliases.v12.0.txt.gz — maps each accession
    to its STRING protein ID (9606.ENSP...). The `UniProt_AC` alias rows cover
    UniProt-keyed graphs (DRYAD, SARS); the `Ensembl*` rows carry each gene's
    `ENSG...` id, so an Ensembl-gene-keyed graph (HuRI) resolves too — see
    `map_accessions_to_ensp`. Verified: 88,155 UniProt_AC mappings / 19,399
    unique STRING proteins for human.
  - local-docs/string/9606.protein.links.detailed.v12.0.txt.gz — 13.7M rows
    (both orderings of every pair present), columns: protein1 protein2
    neighborhood fusion cooccurence coexpression experimental database
    textmining combined_score. Raw scores are 0-1000 integers; rescaled to
    [0, 1] here to match the other rules' score conventions.

A pair with no STRING mapping, or where STRING's links file has no row for
that ENSP pair at all (STRING only lists pairs with some minimum aggregate
evidence), is simply absent from the result — same silent-abstain convention
as every other annotation script in this project. Note this is NOT the same
as "channel value is 0": a listed row can still have channel=0 if the pair's
evidence comes from a *different* channel (see
compute_string_experimental.py's own docstring for how it treats that
distinction for calibration purposes).
"""
from __future__ import annotations

import gzip
from pathlib import Path

ALIASES = Path("local-docs/string/9606.protein.aliases.v12.0.txt.gz")
LINKS = Path("local-docs/string/9606.protein.links.detailed.v12.0.txt.gz")


def map_accessions_to_ensp(accessions: list[str]) -> dict[str, str]:
    """One STRING ENSP per accession (first alias-file occurrence kept, same
    single-canonical-ID convention used elsewhere in this project's annotation
    scripts).

    Resolves two id spaces, so the STRING rule can fire on either a
    UniProt-keyed graph (DRYAD, SARS) or an Ensembl-gene-keyed one (HuRI):
      * UniProt accessions -> ENSP via the file's ``UniProt_AC`` alias rows.
      * Ensembl gene ids (``ENSG...``) -> ENSP via its ``Ensembl*`` alias rows.
        STRING lists a gene's id as an alias on each of that gene's ENSP
        proteins, so a gene with several protein products resolves to whichever
        ENSP the file lists first — arbitrary but deterministic, and acceptable
        for this low-confidence non-interaction signal.

    UniProt matching stays gated on the ``UniProt_AC`` source to avoid
    cross-database alias collisions; an ``ENSG...`` id is globally unique, so
    matching one needs no gate beyond "an Ensembl-sourced row.\""""
    wanted = set(accessions)
    out: dict[str, str] = {}
    with gzip.open(ALIASES, "rt") as fh:
        next(fh)  # header
        for line in fh:
            ensp, alias, source = line.rstrip("\n").split("\t")
            if alias not in wanted or alias in out:
                continue
            if alias.startswith("ENSG"):
                if "Ensembl" not in source:
                    continue
            elif "UniProt_AC" not in source:
                continue
            out[alias] = ensp
    return out


def channel_scores_for_ensp_pairs(ensp_pairs: set[frozenset[str]],
                                   channel: str) -> dict[frozenset[str], float]:
    """Streams the 13.7M-row links file once, keeping only rows whose ENSP
    pair is in `ensp_pairs`, keyed by the requested `channel` column. Each
    pair appears twice (both orderings) with identical scores; either
    occurrence is fine to keep."""
    scores: dict[frozenset[str], float] = {}
    needed = set(ensp_pairs)
    with gzip.open(LINKS, "rt") as fh:
        header = next(fh).split()
        col = header.index(channel)
        for line in fh:
            parts = line.split()
            pair = frozenset((parts[0], parts[1]))
            if pair in needed and pair not in scores:
                scores[pair] = int(parts[col]) / 1000.0
                if len(scores) == len(needed):
                    break
    return scores


def compute(pairs: list[tuple[str, str]], channel: str) -> dict[frozenset[str], float]:
    """Returns {frozenset({acc_a, acc_b}): score in [0, 1]} for `channel`,
    for pairs where both accessions map to a STRING ENSP AND STRING's links
    file has a row for that ENSP pair (see module docstring for why "has a
    row" isn't the same as "channel value > 0")."""
    accessions = sorted({acc for pair in pairs for acc in pair})
    acc_to_ensp = map_accessions_to_ensp(accessions)
    print(f"  mapped {len(acc_to_ensp)}/{len(accessions)} accessions to a STRING protein ID")

    ensp_pair_to_accs: dict[frozenset[str], tuple[str, str]] = {}
    for acc_a, acc_b in pairs:
        ensp_a, ensp_b = acc_to_ensp.get(acc_a), acc_to_ensp.get(acc_b)
        if ensp_a and ensp_b:
            ensp_pair_to_accs[frozenset((ensp_a, ensp_b))] = (acc_a, acc_b)

    ensp_scores = channel_scores_for_ensp_pairs(set(ensp_pair_to_accs), channel)
    scores: dict[frozenset[str], float] = {}
    for ensp_pair, acc_pair in ensp_pair_to_accs.items():
        if ensp_pair in ensp_scores:
            scores[frozenset(acc_pair)] = ensp_scores[ensp_pair]
    print(f"scored {len(scores)}/{len(pairs)} pairs "
          f"({len(ensp_pair_to_accs) - len(scores)} mapped to STRING but had no reported "
          f"{channel} row)")
    return scores


def write_merged(path: Path, new_values: dict[frozenset[str], float]) -> int:
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


def read_pairs_file(path: str | Path) -> list[tuple[str, str]]:
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            a, b = line.split("\t")[:2]
            pairs.append((a, b))
    return pairs
