"""Export the SARS-CoV-2 interactome to the generic `negaverse run --input`
TSV format (u, v, u_type, v_type), so:

    negaverse run --input local-docs/sars/covid.tsv --modality ppi

exercises the same graph as `python -m negaverse.cli` (the built-in SARS demo),
just through the dataset-agnostic loader instead of the bespoke xlsx reader.

Reuses load_sars_cov2_graph() rather than re-parsing the xlsx, so this can
never drift from the real loader's bait/prey/host-host logic.
"""
from __future__ import annotations

from pathlib import Path

from negaverse.io import load_sars_cov2_graph

OUT = Path("local-docs/sars/covid.tsv")


def main() -> None:
    graph = load_sars_cov2_graph()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for u, v in graph.g.edges():
            fh.write(f"{u}\t{v}\t{graph.node_type[u]}\t{graph.node_type[v]}\n")
    print(f"wrote {graph.g.number_of_edges()} edges -> {OUT}")


if __name__ == "__main__":
    main()
