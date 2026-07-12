"""STRING-channel id resolution and end-to-end scoring.

Proves: `map_accessions_to_ensp` resolves BOTH UniProt accessions (DRYAD/SARS
graphs) and Ensembl gene ids (HuRI graph), and that `compute()` keys its output
by the *input* accession — so feeding an ENSG-keyed pair yields an ENSG-keyed
score row, which is what lets `string_low_confidence_non_interaction` fire on
HuRI. Uses tiny synthetic aliases/links files (not the real 88k-row download).

    python -m pytest tests/test_string_channel.py
"""
from __future__ import annotations

import gzip
from pathlib import Path

import scripts.string_channel as sc

# STRING aliases file: string_protein_id<TAB>alias<TAB>source, one per line.
# P1/P2 are UniProt accessions; ENSGAAA/ENSGBBB are Ensembl gene ids that STRING
# lists as aliases on the same two ENSP proteins (under an Ensembl* source).
_ALIASES = "\n".join([
    "#string_protein_id\talias\tsource",
    "9606.ENSP001\tP1\tEnsembl_UniProt_AC",
    "9606.ENSP001\tENSGAAA\tEnsembl_gene",
    "9606.ENSP002\tP2\tBLAST_UniProt_AC",
    "9606.ENSP002\tENSGBBB\tEnsembl_HGNC_ensembl_gene_id",
    # a decoy: an ENSG-looking alias under a non-Ensembl source must be ignored
    "9606.ENSP999\tENSGAAA\tSomeOtherDB",
]) + "\n"

# STRING links.detailed: whitespace-separated, header then rows (both orderings).
_LINKS = "\n".join([
    "protein1 protein2 neighborhood fusion cooccurence coexpression experimental database textmining combined_score",
    "9606.ENSP001 9606.ENSP002 0 0 0 0 120 0 0 120",
    "9606.ENSP002 9606.ENSP001 0 0 0 0 120 0 0 120",
]) + "\n"


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _patch_files(tmp_path, monkeypatch):
    aliases = tmp_path / "aliases.txt.gz"
    links = tmp_path / "links.txt.gz"
    _write_gz(aliases, _ALIASES)
    _write_gz(links, _LINKS)
    monkeypatch.setattr(sc, "ALIASES", aliases)
    monkeypatch.setattr(sc, "LINKS", links)


def test_uniprot_accessions_resolve_to_ensp(tmp_path, monkeypatch):
    """Regression: the original UniProt_AC path is unchanged."""
    _patch_files(tmp_path, monkeypatch)
    assert sc.map_accessions_to_ensp(["P1", "P2"]) == {
        "P1": "9606.ENSP001", "P2": "9606.ENSP002"}


def test_ensembl_gene_ids_resolve_to_ensp(tmp_path, monkeypatch):
    """The fix: HuRI's ENSG nodes resolve via the Ensembl* alias rows, and a
    non-Ensembl ENSG-looking alias (ENSP999 decoy) is not accepted."""
    _patch_files(tmp_path, monkeypatch)
    assert sc.map_accessions_to_ensp(["ENSGAAA", "ENSGBBB"]) == {
        "ENSGAAA": "9606.ENSP001", "ENSGBBB": "9606.ENSP002"}


def test_compute_keys_output_by_input_id_for_both_spaces(tmp_path, monkeypatch):
    """`compute()` returns scores keyed by the *input* accession, so an ENSG
    pair produces an ENSG-keyed row (matching HuRI graph node ids) exactly as a
    UniProt pair produces a UniProt-keyed one. 120/1000 -> 0.12."""
    _patch_files(tmp_path, monkeypatch)

    uni = sc.compute([("P1", "P2")], "experimental")
    assert uni == {frozenset({"P1", "P2"}): 0.12}

    ensg = sc.compute([("ENSGAAA", "ENSGBBB")], "experimental")
    assert ensg == {frozenset({"ENSGAAA", "ENSGBBB"}): 0.12}


def test_unresolvable_accession_silently_absent(tmp_path, monkeypatch):
    """An id with no alias row is dropped (silent-abstain convention), not an
    error — mirrors every other annotation script here."""
    _patch_files(tmp_path, monkeypatch)
    assert sc.map_accessions_to_ensp(["ENSGAAA", "UNKNOWN"]) == {
        "ENSGAAA": "9606.ENSP001"}
    assert sc.compute([("ENSGAAA", "UNKNOWN")], "experimental") == {}
