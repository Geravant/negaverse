"""Assemble a single self-contained HTML dashboard from a run's outputs.

Reads whatever is in `out_dir` — `stats.json` (metrics + validation) and every
`*.png` panel — and writes `out_dir/report.html` with the images base64-embedded
(so the file is portable: open it anywhere, no server, no external files).

    from negaverse.viz import build_report
    build_report("out")                      # after a pipeline run
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

# panel filename -> (title, caption). Unlisted PNGs still get shown, titled by name.
_PANELS = {
    "manifold.png": ("Manifold — the 4 regimes",
                     "Pairs in topology-feature space: positives (the manifold), random negatives "
                     "(far), hard negatives (close but distinguishable), risky negatives (inside)."),
    "separability.png": ("Separability",
                         "Common-neighbour overlap and graph distance: hard negatives sit closer to "
                         "the positives than random negatives do."),
    "confidence_hardness.png": ("Negative regimes: confidence × hardness",
                                "Every emitted negative on negaverse's own axes — eval (safe), train "
                                "(hard), and the risky suspected-false-negative tail."),
    "flag_breakdown.png": ("Flag breakdown",
                           "Why each pair is what it is: provenance flags across the emitted set."),
    "funnel.png": ("Hourglass funnel",
                   "Pairs kept at each stage: candidates → VETO → GRADED → GATED → emitted."),
}
_ORDER = ["manifold.png", "separability.png", "confidence_hardness.png",
          "flag_breakdown.png", "funnel.png"]


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _cards(stats: dict, validation: dict) -> list[tuple[str, str]]:
    c: list[tuple[str, str]] = []
    em = stats.get("emitted", {})
    if em:
        c.append(("emitted negatives", f"{sum(em.values())}  ({em.get('eval',0)} eval / {em.get('train',0)} train)"))
    if "candidates" in stats:
        c.append(("candidate pool", f"{stats['candidates']:,}"))
    if "scored_pool" in stats:
        c.append(("scored (survived VETO)", f"{stats['scored_pool']:,}"))
    if "gated_reviewed" in stats:
        c.append(("gated (LLM reviewed)", str(stats["gated_reviewed"])))
    lk = validation.get("leakage_known_positive")
    if lk is not None:
        c.append(("known-positive leakage", f"{lk}  ✓" if lk == 0 else str(lk)))
    dm = validation.get("degree_match", {})
    if "improvement" in dm:
        c.append(("degree-match (KS vs random)",
                  f"{dm.get('ks_negaverse_vs_positive','?')} vs {dm.get('ks_random_vs_positive','?')}"))
    hs = validation.get("hardness_split", {})
    if hs:
        c.append(("hardness (train / eval)",
                  f"{hs.get('train_mean_hardness','?')} / {hs.get('eval_mean_hardness','?')}"))
    gr = validation.get("gold_recall", {})
    if isinstance(gr, dict) and "golds_in_pool" in gr:
        c.append(("gold negatives in pool", str(gr["golds_in_pool"])))
    lit = validation.get("literature", {})
    if isinstance(lit, dict) and lit.get("status"):
        c.append(("literature (LLM)", lit["status"]))
    return c


def build_report(out_dir: str | Path, title: str = "negaverse", subtitle: str = "") -> Path:
    out_dir = Path(out_dir)
    stats_path = out_dir / "stats.json"
    stats, validation = {}, {}
    if stats_path.exists():
        blob = json.loads(stats_path.read_text())
        stats = blob.get("stats", blob)
        validation = blob.get("validation", {})

    pngs = sorted(out_dir.glob("*.png"))
    ordered = [out_dir / n for n in _ORDER if (out_dir / n).exists()]
    ordered += [p for p in pngs if p not in ordered]

    filters = stats.get("filters", {})
    graph = stats.get("graph", {})
    sub = subtitle or (f"{graph.get('name','')} — {graph.get('nodes','?')} nodes, "
                       f"{graph.get('edges','?')} edges" if graph else "")

    def img_block(p: Path) -> str:
        b64 = base64.b64encode(p.read_bytes()).decode()
        ttl, cap = _PANELS.get(p.name, (p.stem.replace("_", " "), ""))
        return (f'<section class="panel"><h2>{ttl}</h2>'
                f'{f"<p class=cap>{cap}</p>" if cap else ""}'
                f'<img src="data:image/png;base64,{b64}" alt="{ttl}"></section>')

    cards = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                    for k, v in _cards(stats, validation))
    filt = ""
    if filters:
        filt = ('<div class="filters">' + "".join(
            f'<span class="stage">{s.upper()}: {", ".join(filters.get(s, []) or ["—"])}</span>'
            for s in ("veto", "graded", "gated")) + "</div>")
    panels = "".join(img_block(p) for p in ordered) or "<p>No panels found — run the viz first.</p>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — report</title>
<style>
 :root {{ color-scheme: light dark; --bg:#fafafa; --fg:#1a1a1a; --card:#fff; --mut:#666; --line:#e4e4e4; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171a; --fg:#e8e8e8; --card:#1e2126; --mut:#9aa0a6; --line:#2c2f36; }} }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }}
 header {{ padding:28px 24px 8px; }}
 h1 {{ margin:0; font-size:24px; }} .sub {{ color:var(--mut); margin-top:4px; }}
 .wrap {{ max-width:1080px; margin:0 auto; padding:16px 24px 60px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:16px 0; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
 .card .k {{ color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
 .card .v {{ font-size:18px; font-weight:600; margin-top:4px; }}
 .filters {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 4px; }}
 .stage {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:4px 12px; font-size:13px; color:var(--mut); }}
 .panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:18px 0; }}
 .panel h2 {{ margin:0 0 4px; font-size:17px; }} .cap {{ color:var(--mut); margin:0 0 12px; font-size:13.5px; }}
 .panel img {{ width:100%; height:auto; border-radius:6px; }}
</style></head><body>
<header><div class="wrap" style="padding-bottom:0"><h1>{title} — run report</h1>
<div class="sub">{sub}</div></div></header>
<div class="wrap">
 <div class="cards">{cards}</div>
 {filt}
 {panels}
</div></body></html>"""
    p = out_dir / "report.html"
    p.write_text(html)
    return p
