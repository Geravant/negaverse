"""Self-contained interactive-3D HTML report for the external benchmark datasets
(UPNA-PPI, DRYAD) — a rotatable map of pairs coloured by class, in each dataset's
OWN signal space (topology for UPNA, ESM2 for DRYAD), plus an AUROC summary.

These datasets don't carry our compartment/hydrophobicity annotations, so the
SARS/HuRI 3-axis dashboard (`viz.report`) doesn't apply; this renders the honest
per-dataset equivalent. Plotly is inlined from the cached copy (offline/portable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_COLORS = ["#2a9d8f", "#e63946", "#e9c46a", "#4c6ef5", "#adb5bd", "#b5179e"]


def render_3d_report(out_path: str | Path, title: str, subtitle: str,
                     classes: list[dict], axis_labels: tuple[str, str, str],
                     summary_rows: list[tuple[str, str]], caption: str = "",
                     max_points: int = 1500, seed: int = 0) -> Path:
    """classes: [{name, points: Nx3 array-like, color?, hover?: list[str]}].
    Writes a self-contained HTML with an inline Plotly 3D scatter (one trace per
    class) + an AUROC summary table. Returns the path."""
    from .interactive import get_plotly_js
    rng = np.random.default_rng(seed)
    traces = []
    for i, c in enumerate(classes):
        pts = np.asarray(c["points"], dtype=float)
        if pts.size == 0:
            continue
        if len(pts) > max_points:                       # keep the file light
            keep = rng.choice(len(pts), max_points, replace=False)
            pts = pts[keep]
            hover = [c["hover"][k] for k in keep] if c.get("hover") else None
        else:
            hover = c.get("hover")
        traces.append({
            "type": "scatter3d", "mode": "markers",
            "name": f"{c['name']} ({len(pts)})",
            "x": np.round(pts[:, 0], 4).tolist(),
            "y": np.round(pts[:, 1], 4).tolist(),
            "z": np.round(pts[:, 2], 4).tolist(),
            "text": hover, "hoverinfo": "text" if hover else "name",
            "marker": {"size": 2.5, "color": c.get("color", _COLORS[i % len(_COLORS)]),
                       "opacity": 0.7},
        })
    layout = {"scene": {"xaxis": {"title": axis_labels[0]},
                        "yaxis": {"title": axis_labels[1]},
                        "zaxis": {"title": axis_labels[2]}},
              "margin": {"l": 0, "r": 0, "t": 0, "b": 0}, "height": 600,
              "legend": {"x": 0, "y": 1}, "paper_bgcolor": "rgba(0,0,0,0)"}

    rows = "".join(f"<tr><td>{_esc(k)}</td><td class='v'>{_esc(v)}</td></tr>"
                   for k, v in summary_rows)
    lib = get_plotly_js()
    plot = (f'<div id="p3d" style="width:100%;height:600px"></div>'
            f'<script>{lib}</script>'
            f'<script>var _D={{"traces":{json.dumps(traces)},"layout":{json.dumps(layout)}}};'
            'Plotly.newPlot("p3d",_D.traces,_D.layout,{responsive:true,displayModeBar:false});'
            '</script>') if lib else \
           ('<p class="cap"><i>Interactive view needs the Plotly library (fetched once when '
            'built online). Re-run with internet access to enable it.</i></p>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>
:root{{color-scheme:light dark}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;
line-height:1.5;background:#fbfbfd;color:#1d1d1f}}
@media(prefers-color-scheme:dark){{body{{background:#0d0d0f;color:#e8e8ea}}
.panel{{background:#161619!important;border-color:#2a2a2e!important}}}}
.wrap{{max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 2px}}.sub{{color:#86868b;margin:0 0 20px}}
.panel{{background:#fff;border:1px solid #e5e5ea;border-radius:14px;padding:20px;margin:16px 0}}
h2{{font-size:1.05rem;margin:0 0 10px}}
.cap{{color:#86868b;font-size:.9rem;margin:6px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}}
td{{padding:6px 10px;border-bottom:1px solid #e5e5ea}}
@media(prefers-color-scheme:dark){{td{{border-color:#2a2a2e}}}}
td.v{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}}
</style></head><body><div class="wrap">
<h1>{_esc(title)}</h1><p class="sub">{_esc(subtitle)}</p>
<section class="panel"><h2>Separation summary</h2>
<table>{rows}</table></section>
<section class="panel"><h2>Interactive 3D map — drag to rotate, scroll to zoom</h2>
<p class="cap">{caption}</p>{plot}</section>
</div></body></html>"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
