"""Assemble a single self-contained HTML dashboard from a run's outputs.

Reads whatever is in `out_dir` — `stats.json` (metrics + validation) and every
`*.png` panel — and writes `out_dir/report.html` with the images base64-embedded
(so the file is portable: open it anywhere, no server, no external files).

Written for a non-specialist reader: plain-language intro, captions that say what
to look for, and friendly metric labels.

    from negaverse.viz import build_report
    build_report("out")                      # after a pipeline run
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

_INTRO = """
<b>What is this?</b> To learn which proteins work together, machine-learning models
also need examples of proteins that <b>do not</b> work together — called
<b>negatives</b>. The usual shortcut is to pair proteins at random, but that is
unreliable: some random pairs really do interact, we just haven't tested them yet.
<b>negaverse</b> builds better, carefully-checked non-interacting pairs and shows
its reasoning. The panels below show how well it did on this run.
"""

# panel filename -> (title, caption). Captions are plain-language + "look for".
_PANELS = {
    "quadrant.png": (
        "Two lenses at once — does it look real, and can biology allow it?",
        "Left–right = does the pair <i>look</i> like a real interaction in the network. "
        "Up–down = does biology <i>allow</i> it (do the two proteins share a cell compartment, "
        "so they could meet). <b>Look for:</b> "
        "<span style='color:#2a9d8f'>real interactions</span> and "
        "<span style='color:#e63946'>risky pairs</span> in the top-right (look real + could meet — "
        "the risky ones are likely hidden positives); "
        "<span style='color:#e9c46a'>our chosen non-pairs</span> in the bottom-right (look real by "
        "the network, but biology says they can't meet — the strong, useful negatives); "
        "<span style='color:#adb5bd'>random pairs</span> off to the left. This is the picture of "
        "why a hard negative can sit close to positives yet still be told apart."),
    "manifold.png": (
        "Map of protein pairs",
        "A <b>map</b>, like flattening a globe: many network measurements per pair are squeezed "
        "into 2 axes, so what matters is <b>distance</b> — dots near each other are similar pairs, "
        "and the axis numbers themselves have no units. "
        "<b>Look for:</b> the <span style='color:#2a9d8f'>real interacting pairs</span> form a "
        "cloud; <span style='color:#adb5bd'>random guesses</span> land far away (easy to tell "
        "apart); our <span style='color:#e9c46a'>selected non-pairs</span> sit near the real ones "
        "(usefully challenging); and <span style='color:#e63946'>risky picks</span> land right "
        "inside the real cloud — pairs that might secretly interact, so we flag them."),
    "separability.png": (
        "Do our non-pairs look different from random ones?",
        "Two simple network measures. <b>Look for:</b> our selected non-pairs (orange) look more "
        "like the real interacting pairs than plain random guesses (grey) do — that's what makes "
        "them better training examples."),
    "confidence_hardness.png": (
        "Confidence vs. how real-looking each pair is",
        "Each dot is a non-pair we produced. Left–right = how sure we are they don't interact; "
        "up–down = how much they still resemble a real interacting pair. <b>Look for:</b> the "
        "<span style='color:#457b9d'>benchmark set</span> (confident, safe), the "
        "<span style='color:#e9c46a'>training set</span> (deliberately challenging), and the "
        "<span style='color:#e63946'>risky tail</span> we flag as maybe-really-interacting."),
    "flag_breakdown.png": (
        "Provenance flags per emitted pair",
        "The audit trail: which flags each filter attached during scoring. "
        "<code>different_compartment</code> = the two proteins share no GO cellular-component term "
        "(co-localization rule fired). "
        "<code>no_shared_neighbors_low_expected_edge</code> = no common graph neighbours and a "
        "near-zero configuration-model expected-edge count (topology rule). "
        "<code>near_boundary</code> = hardness ≥ 0.90 (top-decile topological closeness to the "
        "positive manifold). "
        "<code>suspected_false_negative</code> = bottom-quantile fused confidence, or the "
        "<b>manifold</b> filter finds the pair as positive-like as a typical real edge in embedding "
        "space — flagged for review, not shipped as a confident negative. "
        "<code>topology_manifold_disagreement</code> = the two independent graph views (network "
        "shape vs. embedding manifold) conflict on this pair, so it's routed to AI review — the "
        "manifold's unique signal lives where it disagrees with topology. Full per-filter sub-scores "
        "+ evidence are in <code>out/negatives.jsonl</code>."),
    "funnel.png": (
        "How pairs were filtered, step by step",
        "We start from many candidate pairs and narrow down. <b>Look for:</b> quick rejects first, "
        "then scoring, then a small expert-reviewed set, ending with the pairs we keep."),
}
_ORDER = ["quadrant.png", "manifold.png", "separability.png", "confidence_hardness.png",
          "flag_breakdown.png", "funnel.png"]


def _lit_phrase(lit: dict) -> str:
    s = (lit or {}).get("status", "")
    if s == "ran":
        n = lit.get("cards", "")
        return f"on — reviewed {n} uncertain pairs" if n != "" else "on"
    if s == "disabled":
        return "off (this run used --no-literature)"
    if s in ("skipped", "no_api_key"):
        return "off (no API key set)"
    return s or "not run"


def _cards(stats: dict, validation: dict) -> list[tuple[str, str, str]]:
    """(label, value, one-line meaning) — plain language."""
    c: list[tuple[str, str, str]] = []
    em = stats.get("emitted", {})
    if em:
        c.append(("Non-interacting pairs produced", str(sum(em.values())),
                  f"{em.get('eval',0)} for a fair benchmark, {em.get('train',0)} harder ones for training"))
    if "candidates" in stats:
        c.append(("Pairs considered", f"{stats['candidates']:,}", "the starting pool of possible non-pairs"))
    rc = stats.get("risky_coverage") or {}
    if rc.get("risky"):
        judged, total, un = rc.get("judged", 0), rc.get("risky", 0), rc.get("unjudged", 0)
        meaning = "risky pairs (possible hidden positives) that got the LLM verdict"
        if un:
            meaning += (f" — {un} still unjudged; raise --literature-k "
                        f"(now {rc.get('gated_cap','?')}) or run --judge-remaining")
        c.append(("Risky pairs AI-reviewed", f"{judged} of {total}" + (" ✓" if not un else ""),
                  meaning))
    elif "gated_reviewed" in stats:
        c.append(("Pairs sent for AI review", str(stats["gated_reviewed"]),
                  "only the most uncertain pairs, to save cost"))
    lk = validation.get("leakage_known_positive")
    if lk is not None:
        c.append(("Real interactions that slipped in", f"{lk}" + (" ✓" if lk == 0 else ""),
                  "should be 0 — no known interaction is mislabelled as a non-pair"))
    dm = validation.get("degree_match", {})
    if "improvement" in dm:
        c.append(("Benchmark fairness",
                  f"{dm.get('ks_negaverse_vs_positive','?')} vs {dm.get('ks_random_vs_positive','?')} random",
                  "how well-matched to real pairs (lower is better) — ours beats random"))
    hs = validation.get("hardness_split", {})
    if hs:
        c.append(("Challenge level (training vs benchmark)",
                  f"{hs.get('train_mean_hardness','?')} vs {hs.get('eval_mean_hardness','?')}",
                  "training pairs are deliberately harder (closer to real interactions)"))
    gr = validation.get("gold_recall", {})
    if isinstance(gr, dict) and "golds_in_pool" in gr:
        c.append(("Verified non-interactions found", str(gr["golds_in_pool"]),
                  "overlap with an external gold-standard list"))
    lit = validation.get("literature")
    if isinstance(lit, dict) and lit.get("status"):
        c.append(("AI literature review", _lit_phrase(lit),
                  "an LLM double-checks the most uncertain pairs"))
    return c


def _interactive_panel(out_dir: Path) -> str:
    """Inline Plotly 3D scatter (portable) — three meaningful axes, hover + rotate."""
    j = out_dir / "interactive3d.json"
    if not j.exists():
        return ""
    from .interactive import get_plotly_js
    lib = get_plotly_js()
    head = ('<section class="panel"><h2>Interactive 3D map — drag to rotate, scroll to zoom, '
            'hover a dot</h2><p class="cap">Three independent lenses:<br>'
            '&bull; <b>x — looks like a real interaction:</b> judged only from the <i>shape</i> of '
            'the known interaction network. Two proteins score high when they sit where real '
            'partners usually sit — linked through short chains of shared partners (friend-of-a-'
            'friend patterns that interacting proteins tend to show) — even if no direct link is '
            'recorded between them. High = the network thinks they <i>should</i> interact.<br>'
            '&bull; <b>y — biology allows it:</b> do the two proteins share a cell compartment, so '
            'they could physically meet (from GO cellular-component).<br>'
            '&bull; <b>z — chemistry match:</b> how similar their surface hydrophobicity is.<br>'
            'Colours are the four regimes; hover any point for the two proteins and why it is '
            'flagged. Pairs whose proteins lack compartment or hydrophobicity data (e.g. viral '
            'proteins in the SARS-CoV-2 graph) sit on the base plane (y or z = 0) and are marked '
            '<i>no compartment / no hydrophobicity data</i> in the hover — they are still placed '
            'on the network-shape axis, which needs no annotation. Points are given a tiny jitter '
            'so many identical pairs (e.g. all emitted negatives at the topology floor) show as a '
            'visible cloud instead of one overlapping dot.</p>')
    if not lib:
        return (head + '<p class="cap"><i>The interactive view needs the Plotly library, which '
                'is fetched once when the report is built online. Re-run the dashboard with an '
                'internet connection to enable it.</i></p></section>')
    data = j.read_text()
    return (head + '<div id="p3d" style="width:100%;height:580px"></div>'
            f'<script>{lib}</script>'
            f'<script>var _D={data};Plotly.newPlot("p3d",_D.traces,_D.layout,'
            '{responsive:true,displayModeBar:false});</script></section>')


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _lit_details(out_dir: Path) -> str:
    """Collapsible section showing the actual LLM literature checks, if present."""
    p = out_dir / "literature_cards.json"
    if not p.exists():
        return ""
    try:
        cards = json.loads(p.read_text())
    except Exception:
        return ""
    if not cards:
        return ""
    _V = {"safe_negative": "#2a9d8f", "uncertain": "#e9a13b",
          "suspected_false_negative": "#e63946"}
    rows = []
    for c in cards:
        col = _V.get(c.get("verdict", ""), "#888")
        ev = "".join(f"<li>{_esc(e)}</li>" for e in c.get("evidence", []))
        vc = ", ".join(f"{k}: {v}" for k, v in (c.get("vote_counts") or {}).items())
        rows.append(
            f'<details class="lit"><summary>'
            f'<code>{_esc(c.get("u"))} × {_esc(c.get("v"))}</code> — '
            f'<b style="color:{col}">{_esc(c.get("verdict"))}</b> '
            f'<span class="mut">(agreement {c.get("agreement","?")}; votes — {_esc(vc)})</span>'
            f'</summary>'
            f'<p class="rat">{_esc(c.get("rationale",""))}</p>'
            f'{f"<ul>{ev}</ul>" if ev else ""}'
            f'<div class="mut sm">model: {_esc(c.get("model",""))}</div>'
            f'</details>')
    return (
        '<section class="panel"><h2>AI literature review — the actual checks</h2>'
        '<p class="cap">The LLM read the most uncertain pairs and returned a structured verdict '
        'with a rationale and supporting points. <b>“uncertain”</b> means it would not confidently '
        'call the pair a safe non-interaction (best-of-N majority vote; low agreement = genuinely '
        'undecided). Click a pair to expand.</p>'
        f'<details class="litwrap"><summary>Show {len(cards)} reviewed pairs</summary>'
        + "".join(rows) + '</details></section>')


def build_report(out_dir: str | Path, title: str = "negaverse", subtitle: str = "") -> Path:
    out_dir = Path(out_dir)
    stats_path = out_dir / "stats.json"
    stats, validation = {}, {}
    if stats_path.exists():
        blob = json.loads(stats_path.read_text())
        stats = blob.get("stats", blob)
        validation = blob.get("validation", {})

    # only the panels this run produces — keep unrelated experiment PNGs
    # (e.g. the UPNA/DRYAD validation panels) out of the demo dashboard.
    ordered = [out_dir / n for n in _ORDER if (out_dir / n).exists()]

    graph = stats.get("graph", {})
    sub = subtitle or (f"dataset: {graph.get('name','')} — {graph.get('nodes','?')} proteins, "
                       f"{graph.get('edges','?')} known interactions" if graph else "")

    def img_block(p: Path) -> str:
        b64 = base64.b64encode(p.read_bytes()).decode()
        ttl, cap = _PANELS.get(p.name, (p.stem.replace("_", " ").title(), ""))
        return (f'<section class="panel"><h2>{ttl}</h2>'
                f'{f"<p class=cap>{cap}</p>" if cap else ""}'
                f'<img src="data:image/png;base64,{b64}" alt="{ttl}"></section>')

    cards = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="h">{h}</div></div>'
        for k, v, h in _cards(stats, validation))
    panels = "".join(img_block(p) for p in ordered) or "<p>No panels found — run the viz first.</p>"
    interactive = _interactive_panel(out_dir)
    lit = _lit_details(out_dir)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — report</title>
<style>
 :root {{ color-scheme: light dark; --bg:#fafafa; --fg:#1a1a1a; --card:#fff; --mut:#666; --line:#e4e4e4; --accent:#2a9d8f; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171a; --fg:#e8e8e8; --card:#1e2126; --mut:#9aa0a6; --line:#2c2f36; }} }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }}
 .wrap {{ max-width:1080px; margin:0 auto; padding:8px 24px 60px; }}
 header {{ padding:30px 24px 0; }}
 h1 {{ margin:0; font-size:26px; }} .sub {{ color:var(--mut); margin-top:4px; }}
 .intro {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent);
           border-radius:10px; padding:16px 18px; margin:18px 0 6px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:16px 0; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:13px 15px; }}
 .card .k {{ color:var(--fg); font-size:13px; font-weight:600; }}
 .card .v {{ font-size:22px; font-weight:700; margin:3px 0; color:var(--accent); }}
 .card .h {{ color:var(--mut); font-size:12.5px; line-height:1.4; }}
 .panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:20px 0; }}
 .panel h2 {{ margin:0 0 6px; font-size:18px; }} .cap {{ color:var(--mut); margin:0 0 12px; font-size:14px; }}
 .panel img {{ width:100%; height:auto; border-radius:6px; }}
 code {{ background:rgba(127,127,127,.14); padding:1px 5px; border-radius:4px; font-size:12.5px; }}
 .mut {{ color:var(--mut); }} .sm {{ font-size:12px; }} .rat {{ margin:8px 0; }}
 .litwrap > summary {{ cursor:pointer; font-weight:600; padding:6px 0; }}
 details.lit {{ border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin:8px 0; }}
 details.lit > summary {{ cursor:pointer; list-style:none; }}
 details.lit ul {{ margin:6px 0 4px; padding-left:20px; }} details.lit li {{ margin:2px 0; font-size:13.5px; }}
</style></head><body>
<header><div class="wrap" style="padding-bottom:0"><h1>{title} — results</h1>
<div class="sub">{sub}</div></div></header>
<div class="wrap">
 <div class="intro">{_INTRO}</div>
 <div class="cards">{cards}</div>
 {interactive}
 {panels}
 {lit}
</div></body></html>"""
    p = out_dir / "report.html"
    p.write_text(html)
    return p
