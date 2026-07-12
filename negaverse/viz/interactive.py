"""Interactive 3D scatter (Plotly) for the dashboard — three *meaningful*,
independent axes so rotating/hovering actually explains the pairs:

  x = looks like a real interaction   (network proximity, topology risk)
  y = biology allows it               (shared subcellular compartments)
  z = chemistry match                 (hydrophobicity similarity)

colour = regime (real / random / our hard / risky), hover = the two protein ids
+ why the pair is flagged, and — for risky pairs the LLM reviewed — the model's
verdict and reasoning inline (read straight off the dot, no scrolling to the
cards section). Plotly is inlined into report.html (portable/offline) from a
cached copy; first build fetches it once.
"""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import numpy as np

from ..graph import TypedInteractionGraph

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_CACHE = Path("local-docs/.cache/plotly.min.js")


def _wrap(text: str, width: int = 64) -> str:
    """Soft-wrap a rationale into <br>-separated lines for a readable Plotly hover
    box (Plotly does not wrap long hover strings on its own)."""
    import textwrap
    if not text:
        return ""
    return "<br>".join(textwrap.wrap(str(text), width=width)) or ""


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
                   n_ref: int = 400, x_axis: "tuple | None" = None) -> dict:
    """x_axis lets a caller replace the default topology x-axis with another
    "looks-real" lens — e.g. sequence/ESM2 resemblance on graphs too sparse for
    topology (DRYAD). Pass (fn(u,v)->float|None, title, missing_note)."""
    from ..streams import TopologyFilter
    from ..io.annotations import build_annotation_table
    from .plots import _random_nonedges
    ann = build_annotation_table()
    rng = np.random.default_rng(seed)

    if x_axis is not None:
        x_fn, x_title, x_missing = x_axis
    else:
        tf = TopologyFilter(); tf.fit(graph)

        def x_fn(u, v):
            s = tf.score(graph, u, v); ev = s.evidence or {}
            return float(ev.get("risk", 0.0)) if s.value is not None else None

        x_title, x_missing = "looks real (network shape)", "no network-shape data"

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
    # LLM literature verdicts (when the gated stream ran) keyed order-invariantly,
    # so hovering a risky pair reads the model's actual reasoning, not just its flag.
    verdicts = {}
    for r in records:
        g = (getattr(r, "provenance", None) or {}).get("gated", {}).get("literature")
        if g and g.get("verdict"):
            verdicts[frozenset((r.u, r.v))] = g
    hard = [(r.u, r.v) for r in records if r.mode == "train" and "suspected_false_negative" not in r.flags]
    risky = [(r.u, r.v) for r in records if "suspected_false_negative" in r.flags]
    rand = _random_nonedges(graph, n_ref, seed)
    cats = [("real interactions", edges, "#2a9d8f"),
            ("random non-pairs", rand, "#adb5bd"),
            ("our chosen non-pairs", hard, "#e9c46a"),
            ("risky — may interact", risky, "#e63946")]

    # The x-axis (topology risk) is computable for ANY pair straight from the
    # graph; only y (compartments) and z (hydrophobicity) need external
    # annotation. On graphs where one partner class is unannotated — e.g. the
    # viral proteins in the SARS-CoV-2 viral-host graph — dropping a pair for a
    # missing y/z would silently erase every positive and every emitted negative
    # (they all touch a viral protein), leaving only annotated host-host randoms.
    # So keep the pair, park the missing lens on the base plane (0.0), and say so
    # in the hover — the point stays visible and honest on the axes that do apply.
    _BASE = 0.0
    # When a whole regime is degenerate on these axes — e.g. SARS emitted
    # negatives are all at the topology floor (x=0.02) with viral y=z on the base
    # plane — 400 identical points collapse to one occluded dot and the colour
    # vanishes. A tiny seeded jitter spreads exact-overlaps into a visible cloud
    # without touching the read (positives sit ~0.6 on x, negatives ~0.02).
    jit = np.random.default_rng(seed)

    def _jitter(vals):
        return [round(v + float(jit.normal(0.0, 0.006)), 4) for v in vals]

    traces = []
    for name, pairs, col in cats:
        xs, ys, zs, txt = [], [], [], []
        for u, v in pairs:
            x, y, z = x_fn(u, v), comp(u, v), hyd(u, v)
            missing = []
            if x is None:
                # On the default topology axis a None is rare → park it on the
                # floor. On an alternate axis (e.g. DRYAD's ESM2 model) a None
                # means "no score", and parking it at 0 would fake a "looks
                # non-real" reading — so drop the pair instead.
                if x_axis is not None:
                    continue
                x, _m = _BASE, missing.append(x_missing)
            if y is None:
                y, _m = _BASE, missing.append("no compartment data")
            if z is None:
                z, _m = _BASE, missing.append("no hydrophobicity data")
            xs.append(round(x, 3)); ys.append(round(y, 3)); zs.append(round(z, 3))
            fl = list(flagmap.get((u, v)) or flagmap.get((v, u)) or [])
            note = "; ".join(fl + missing)
            hover = f"<b>{u} × {v}</b>" + (f"<br>{note}" if note else "")
            g = verdicts.get(frozenset((u, v)))
            if g:                                    # LLM read this pair — show its verdict
                agr = g.get("agreement")
                agr_s = f" · agreement {agr:.0%}" if isinstance(agr, (int, float)) else ""
                hover += (f"<br>———<br><b>LLM literature verdict: {g['verdict']}</b>{agr_s}"
                          f"<br>{_wrap(g.get('rationale', ''))}")
            txt.append(hover)
        traces.append({"type": "scatter3d", "mode": "markers",
                       "name": f"{name} ({len(xs)})",
                       "x": _jitter(xs), "y": _jitter(ys), "z": _jitter(zs),
                       "text": txt, "hoverinfo": "text",
                       "marker": {"size": 3, "color": col, "opacity": 0.75}})
    layout = {"scene": {"xaxis": {"title": x_title},
                        "yaxis": {"title": "biology allows it (compartments)"},
                        "zaxis": {"title": "chemistry match (hydrophobicity)"}},
              "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
              "legend": {"x": 0, "y": 1}, "height": 580,
              "paper_bgcolor": "rgba(0,0,0,0)"}
    return {"traces": traces, "layout": layout}
