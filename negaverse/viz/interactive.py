"""Interactive 3D scatter (Plotly) for the dashboard — three *meaningful*,
independent axes so rotating/hovering actually explains the pairs:

  x = looks like a real interaction   (network proximity, topology risk)
  y = biology allows it               (shared subcellular compartments)
  z = chemistry match                 (hydrophobicity similarity)

colour = regime (real / random / our hard / risky), hover = the two protein ids
+ why the pair is flagged. Plotly is inlined into report.html (portable/offline)
from a cached copy; first build fetches it once.
"""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import numpy as np

from ..graph import TypedInteractionGraph

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_CACHE = Path("local-docs/.cache/plotly.min.js")


def get_plotly_js() -> str | None:
    """The Plotly library source (cached under local-docs/, fetched once)."""
    if _CACHE.exists():
        return _CACHE.read_text()
    try:
        ctx = ssl._create_unverified_context()
        data = urllib.request.urlopen(PLOTLY_URL, timeout=45, context=ctx).read().decode()
    except Exception:
        return None
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(data)
    return data


def compute_traces(graph: TypedInteractionGraph, records, seed: int = 0,
                   n_ref: int = 400) -> dict:
    from ..streams import TopologyFilter
    from ..io.annotations import build_annotation_table
    from .plots import _random_nonedges
    tf = TopologyFilter(); tf.fit(graph)
    ann = build_annotation_table()
    rng = np.random.default_rng(seed)

    def risk(u, v):
        s = tf.score(graph, u, v); ev = s.evidence or {}
        return float(ev.get("risk", 0.0)) if s.value is not None else 0.0

    def comp(u, v):
        cu, cv = ann.get(u, {}).get("compartments"), ann.get(v, {}).get("compartments")
        if not cu or not cv:
            return None
        un = len(cu | cv)
        return len(cu & cv) / un if un else None

    def hyd(u, v):
        hu, hv = ann.get(u, {}).get("surface_hydrophobicity"), ann.get(v, {}).get("surface_hydrophobicity")
        return None if hu is None or hv is None else 1.0 - abs(hu - hv)

    edges = [tuple(e) for e in graph.g.edges()]
    if len(edges) > n_ref:
        edges = [edges[i] for i in rng.choice(len(edges), n_ref, replace=False)]
    flagmap = {(r.u, r.v): r.flags for r in records}
    hard = [(r.u, r.v) for r in records if r.mode == "train" and "suspected_false_negative" not in r.flags]
    risky = [(r.u, r.v) for r in records if "suspected_false_negative" in r.flags]
    rand = _random_nonedges(graph, n_ref, seed)
    cats = [("real interactions", edges, "#2a9d8f"),
            ("random non-pairs", rand, "#adb5bd"),
            ("our chosen non-pairs", hard, "#e9c46a"),
            ("risky — may interact", risky, "#e63946")]

    traces = []
    for name, pairs, col in cats:
        xs, ys, zs, txt = [], [], [], []
        for u, v in pairs:
            y, z = comp(u, v), hyd(u, v)
            if y is None or z is None:
                continue
            xs.append(round(risk(u, v), 3)); ys.append(round(y, 3)); zs.append(round(z, 3))
            fl = flagmap.get((u, v)) or flagmap.get((v, u)) or []
            txt.append(f"{u} × {v}" + (f"<br>{'; '.join(fl)}" if fl else ""))
        traces.append({"type": "scatter3d", "mode": "markers",
                       "name": f"{name} ({len(xs)})", "x": xs, "y": ys, "z": zs,
                       "text": txt, "hoverinfo": "text",
                       "marker": {"size": 3, "color": col, "opacity": 0.75}})
    layout = {"scene": {"xaxis": {"title": "looks real (network shape)"},
                        "yaxis": {"title": "biology allows it (compartments)"},
                        "zaxis": {"title": "chemistry match (hydrophobicity)"}},
              "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
              "legend": {"x": 0, "y": 1}, "height": 580,
              "paper_bgcolor": "rgba(0,0,0,0)"}
    return {"traces": traces, "layout": layout}
