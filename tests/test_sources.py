"""External positive sources -> KnownPositiveVeto (union-of-sources exclusion).

    python -m tests.test_sources
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_positive_sources
from negaverse.streams import KnownPositiveVeto


def _graph():
    return TypedInteractionGraph.from_edges(
        [("a", "b")], {n: "protein" for n in "abcd"},
        admissible_types=[("protein", "protein")], name="toy")


def _manifest(d: Path, pairs_text: str) -> Path:
    (d / "src.tsv").write_text(pairs_text)
    man = d / "sources.yaml"
    man.write_text(yaml.safe_dump([{"name": "t", "modality": "ppi",
                                    "path": str(d / "src.tsv"), "id_space": "uniprot"}]))
    return man


def test_loads_and_restricts_to_nodes():
    with tempfile.TemporaryDirectory() as td:
        man = _manifest(Path(td), "a\tc\n# comment\nd b\nX Y\n")   # a-c, d-b in graph; X-Y not
        pairs, report = load_positive_sources(man, restrict_to=set("abcd"))
        assert frozenset(("a", "c")) in pairs and frozenset(("d", "b")) in pairs
        assert frozenset(("X", "Y")) not in pairs          # restricted out
        assert report["loaded"]["t"] == 2


def test_veto_removes_external_positive():
    with tempfile.TemporaryDirectory() as td:
        man = _manifest(Path(td), "a c\nd b\n")
        g = _graph()
        f = KnownPositiveVeto(sources_path=str(man))
        f.fit(g)
        assert f.score(g, "a", "c").veto        # documented elsewhere -> vetoed
        assert f.score(g, "d", "b").veto
        assert f.score(g, "a", "b").veto        # real graph edge -> vetoed
        assert not f.score(g, "a", "d").veto     # neither -> kept


def test_missing_files_are_graceful():
    with tempfile.TemporaryDirectory() as td:
        man = Path(td) / "sources.yaml"
        man.write_text(yaml.safe_dump([{"name": "absent", "modality": "ppi",
                                        "path": f"{td}/nope.tsv", "id_space": "uniprot"}]))
        pairs, report = load_positive_sources(man)
        assert pairs == set() and report["missing"] == ["absent"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
