"""Load the SARS-CoV-2 host-pathogen interactome as a typed graph.

Source: local-docs/.../Network_Table.xlsx (Gordon et al.). Bait-Prey rows are
positive interactions. Viral baits (nsp*/orf*/E/M/N/S) give viral-host edges;
rows whose bait is a human gene are host-host PPIs.

Two subtleties handled here:
  * ID reconciliation — hosts are UniProt as preys but gene *symbols* as
    host-host baits, so we remap symbols to UniProt via the table's own
    PreyGeneName<->PreyUniprotAcc dictionary.
  * candidate space vs signal — we keep the host-host PPI edges in the graph so
    the topological stream has structure to work with, but restrict candidate
    generation to viral-host non-edges (the space negatives are wanted in).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from ..graph import TypedInteractionGraph

DEFAULT_PATH = (
    "local-docs/xlsxUploads_21bba1e9-b17f-43e3-8708-4e5e12ee0591_"
    "sars-cov2-spreadsheets/Network_Table.xlsx"
)


def _is_viral(bait: str) -> bool:
    """SARS-CoV-2 baits are nsp1-16, orf3a/3b/6/7a/8/9b/9c/10, and E/M/N/S.
    Everything else in the Bait column is a human protein (host-host rows)."""
    b = bait.strip()
    return b.lower().startswith(("nsp", "orf")) or b in {"E", "M", "N", "S"}


def load_sars_cov2_graph(
    path: str | Path = DEFAULT_PATH,
    include_host_host: bool = True,
) -> TypedInteractionGraph:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Network_Table"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    hdr = list(rows[0])
    bi = hdr.index("Bait")
    pi = hdr.index("PreyUniprotAcc")
    gi = hdr.index("PreyGeneName")

    data = [r for r in rows[1:] if r[bi] is not None and r[pi] is not None]
    # symbol -> UniProt, so host-host baits (gene symbols) join the UniProt space
    sym2acc = {str(r[gi]).strip(): str(r[pi]).strip()
               for r in data if r[gi] is not None}

    edges: list[tuple[str, str]] = []
    node_type: dict[str, str] = {}
    n_vh = n_hh = 0
    for r in data:
        bait = str(r[bi]).strip()
        prey = str(r[pi]).strip()
        if _is_viral(bait):
            node_type[bait] = "viral"
            node_type.setdefault(prey, "host")
            edges.append((bait, prey))
            n_vh += 1
        elif include_host_host:
            bait_acc = sym2acc.get(bait, bait)   # fall back to symbol if unmapped
            node_type.setdefault(bait_acc, "host")
            node_type.setdefault(prey, "host")
            if bait_acc != prey:
                edges.append((bait_acc, prey))
                n_hh += 1

    return TypedInteractionGraph.from_edges(
        edges, node_type, admissible_types=[("viral", "host")],
        name="sars-cov2-viral-host",
        meta={"source": str(path), "viral_host_edges": n_vh, "host_host_edges": n_hh},
    )
