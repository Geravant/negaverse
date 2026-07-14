"""Matplotlib panels for the Phase-1 demo. No display needed (Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..graph import TypedInteractionGraph

_C = {"positive": "#2a9d8f", "random": "#adb5bd", "hard": "#e76f51"}

# plain-language names for a non-specialist audience
_NAME = {"positive": "real interactions", "random": "random non-pairs",
         "hard": "our chosen non-pairs"}
# provenance flags -> plain phrases
_FLAG = {
    "different_compartment": "different part of the cell (can't meet)",
    "near_boundary": "sits close to real interactions",
    "suspected_false_negative": "risky — might really interact",
    "easy_negative": "clearly unrelated (easy)",
    "no_shared_neighbors_low_expected_edge": "no shared partners in the network",
    "topology_manifold_disagreement": "the two graph views disagree → sent to review",
}


def _flag_label(f: str) -> str:
    return _FLAG.get(f, f.replace("_", " "))


# --- structural measures ------------------------------------------------
def _common_neighbors(g: nx.Graph, pairs) -> np.ndarray:
    adj = {n: set(g[n]) for n in {x for p in pairs for x in p}}
    return np.array([len(adj.get(u, set()) & adj.get(v, set())) for u, v in pairs])


def _shortest_paths(g: nx.Graph, pairs, cutoff: int = 6) -> np.ndarray:
    out = []
    for u, v in pairs:
        if u not in g or v not in g:
            out.append(cutoff + 1)
            continue
        try:
            d = nx.shortest_path_length(g, u, v)
            out.append(min(d, cutoff + 1))
        except nx.NetworkXNoPath:
            out.append(cutoff + 1)
    return np.array(out)


def plot_separability(graph: TypedInteractionGraph, positives, random_neg, hard_neg,
                      out_path: str | Path, random_label: str | None = None) -> Path:
    """Show negaverse's hard negatives sit *between* positives and random
    negatives on structural axes — i.e. they are the topologically positive-like
    (harder) negatives the tool is meant to select.

    random_label: when the "random" slot actually holds a dataset's own gold
    negatives (DRYAD/UPNA) rather than freshly-generated random pairs, override
    the legend/title text so the plot doesn't call them "random" when they aren't."""
    g = graph.g
    sets = {"positive": positives, "random": random_neg, "hard": hard_neg}
    names = dict(_NAME)
    if random_label:
        names["random"] = random_label

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # panel 1: common-neighbour count (clipped for readability)
    clip = 12
    for name, pairs in sets.items():
        if not pairs:
            continue
        cn = np.clip(_common_neighbors(g, pairs), 0, clip)
        bins = np.arange(0, clip + 2) - 0.5
        ax1.hist(cn, bins=bins, density=True, histtype="step", linewidth=2,
                 color=_C[name], label=f"{names[name]} (n={len(pairs)})")
    ax1.set_xlabel("number of shared partner proteins")
    ax1.set_ylabel("share of pairs")
    ax1.set_title("How many partners the two proteins share")
    ax1.legend()

    # panel 2: shortest-path length distribution (proportions)
    labels = ["1", "2", "3", "4", "5", "6", "≥7"]
    width, offsets = 0.26, {"positive": -0.26, "random": 0.0, "hard": 0.26}
    for name, pairs in sets.items():
        if not pairs:
            continue
        sp = _shortest_paths(g, pairs)
        counts = np.array([(sp == d).mean() for d in range(1, 8)])
        ax2.bar(np.arange(7) + offsets[name], counts, width=width,
                color=_C[name], label=names[name])
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("steps apart in the interaction network")
    ax2.set_ylabel("share of pairs")
    ax2.set_title("How far apart the two proteins are")
    ax2.legend()

    fig.suptitle(f"Our chosen non-pairs look more like real interactions than "
                 f"{names['random']} do", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_funnel(stats: dict, out_path: str | Path) -> Path:
    """The hourglass funnel: how many candidates each stage keeps."""
    cand = stats.get("candidates", 0)
    vetoed = stats.get("vetoed", 0)
    scored = stats.get("scored_pool", 0)
    gated = stats.get("gated_reviewed", 0)
    emitted = sum(stats.get("emitted", {}).values())
    stages = [
        ("candidate pairs", cand),
        ("kept after quick reject", cand - vetoed),
        ("scored", scored),
        ("AI-reviewed", gated),
        ("final non-pairs kept", emitted),
    ]
    labels = [s for s, _ in stages]
    values = [v for _, v in stages]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, values, color="#457b9d", height=0.6)
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {v:,}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("number of pairs")
    ax.set_title("How pairs were filtered, step by step")
    ax.margins(x=0.15)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _random_nonedges(graph: TypedInteractionGraph, n: int, seed: int):
    """Random non-interacting pairs for the 'random baseline' regime, **type-matched**
    to the graph's admissible type-space. On a viral–host graph the positives (and our
    emitted negatives) are all host–viral, so an untyped uniform sample — 91% human ⇒
    mostly host–host — would be an unmatched population that spuriously spreads on axes
    where the matched type lacks data (e.g. viral proteins have no compartment). Honoring
    `admissible_types` keeps random on the same footing as the positives. On a single-type
    graph (HuRI/DRYAD, all protein–protein) every pair is admissible, so behaviour is
    unchanged."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.g.nodes())
    N = len(nodes)
    adm = getattr(graph, "admissible_types", None)
    nt = getattr(graph, "node_type", {}) or {}

    def _admissible(a: str, b: str) -> bool:
        if adm is None:
            return True
        return frozenset((nt.get(a), nt.get(b))) in adm

    out, seen = [], set()
    tries, cap = 0, n * 120 + 1000
    while len(out) < n and tries < cap:
        tries += 1
        a, b = nodes[rng.integers(N)], nodes[rng.integers(N)]
        if a == b or graph.g.has_edge(a, b) or not _admissible(a, b):
            continue
        k = frozenset((a, b))
        if k in seen:
            continue
        seen.add(k)
        out.append((a, b))
    return out


def _stratified_sample(graph: TypedInteractionGraph, pairs: list[tuple[str, str]],
                       n: int, seed: int, n_strata: int = 10) -> list[tuple[str, str]]:
    """Sample n pairs from `pairs`, stratified by degree-sum rank (equal-COUNT
    buckets, not equal-width — robust to the power-law-heavy degree
    distributions real PPI graphs have) instead of a plain uniform draw.
    A large population (e.g. DRYAD's 3,000 edges or UPNA's ~3M gold
    negatives) downsampled by plain `rng.choice` risks a visually skewed
    sample purely by chance — overrepresenting one structural regime
    (all-hub or all-peripheral pairs) and misleading the reader about where
    the FULL population actually sits. Stratifying keeps the displayed
    sample's degree-sum spread representative of the real one."""
    if len(pairs) <= n:
        return list(pairs)
    rng = np.random.default_rng(seed)
    deg = dict(graph.g.degree())
    degsum = np.array([deg.get(u, 0) + deg.get(v, 0) for u, v in pairs], dtype=float)
    ranks = np.argsort(np.argsort(degsum))            # 0..len-1, ties broken by input order
    strata = (ranks * n_strata) // len(pairs)
    out: list[tuple[str, str]] = []
    for s in range(n_strata):
        idx = np.where(strata == s)[0]
        if len(idx) == 0:
            continue
        k = min(len(idx), max(1, round(n * len(idx) / len(pairs))))
        out.extend(pairs[i] for i in rng.choice(idx, size=k, replace=False))
    if len(out) > n:                                  # trim rounding overshoot
        idx = rng.choice(len(out), size=n, replace=False)
        out = [out[i] for i in idx]
    elif len(out) < n:                                # pad rounding shortfall
        chosen = {frozenset(p) for p in out}
        remaining = [p for p in pairs if frozenset(p) not in chosen]
        rng.shuffle(remaining)
        out.extend(remaining[:n - len(out)])
    return out


def _is_risky(r) -> bool:
    return "suspected_false_negative" in r.flags


def plot_quadrant(graph: TypedInteractionGraph, records, out_path: str | Path,
                  seed: int = 0, n_ref: int = 500,
                  gold_negatives: list[tuple[str, str]] | None = None) -> Path:
    """Two INDEPENDENT lenses at once (Lucy's three-regime framing):
      x = "looks like a real interaction"  — network proximity (topology risk)
      y = "biology says they can interact" — shared subcellular compartments
    A pair can look real by the network yet be biologically impossible (bottom-
    right) — those are the strong hard negatives; ones that look real AND could
    co-locate (top-right, mixed with positives) are the risky suspected positives;
    random pairs sit far left. Needs compartment annotations (rich on HuRI).

    gold_negatives: a dataset's own labelled negative benchmark (DRYAD/UPNA),
    plotted INSTEAD of freshly-generated random pairs when given — a real
    external negative set is a more meaningful comparison than a synthetic one
    for datasets that ship one."""
    from ..streams import TopologyFilter
    from ..io.annotations import build_annotation_table
    tf = TopologyFilter(); tf.fit(graph)
    ann = build_annotation_table()
    rng = np.random.default_rng(seed)

    def _risk(u, v):
        s = tf.score(graph, u, v); ev = s.evidence or {}
        return float(ev.get("risk", 0.0)) if s.value is not None else 0.0

    def _biocompat(u, v):
        cu, cv = ann.get(u, {}).get("compartments"), ann.get(v, {}).get("compartments")
        if not cu or not cv:
            return None                       # unannotated -> can't place on biology axis
        union = len(cu | cv)
        return len(cu & cv) / union if union else None

    edges = [tuple(e) for e in graph.g.edges()]
    edges = _stratified_sample(graph, edges, n_ref, seed)
    hard = [(r.u, r.v) for r in records if r.mode == "train" and not _is_risky(r)]
    risky = [(r.u, r.v) for r in records if _is_risky(r)]
    if gold_negatives:
        rand, rand_label = _stratified_sample(graph, gold_negatives, n_ref, seed), "dataset gold negatives"
    else:
        rand, rand_label = _random_nonedges(graph, n_ref, seed), "random non-pairs"
    cats = [("real interactions", edges, "#2a9d8f", 0.45),
            (rand_label, rand, "#adb5bd", 0.4),
            ("our chosen non-pairs", hard, "#e9c46a", 0.75),
            ("risky — may interact", risky, "#e63946", 0.9)]

    fig, ax = plt.subplots(figsize=(8, 6.6))
    total = 0
    for name, pairs, col, a in cats:
        xs, ys = [], []
        for u, v in pairs:
            y = _biocompat(u, v)
            if y is None:
                continue
            xs.append(_risk(u, v)); ys.append(y)
        if xs:
            yj = np.array(ys) + rng.uniform(-0.008, 0.008, len(ys))   # jitter the y=0 pile
            ax.scatter(xs, yj, s=16, alpha=a, color=col, label=f"{name} (n={len(xs)})")
            total += len(xs)

    ax.set_xlim(-0.02, max(0.02, ax.get_xlim()[1]))
    ax.axhline(0.05, color="#999", lw=0.8, ls="--")
    xm = 0.5 * (ax.get_xlim()[1])
    ax.axvline(xm, color="#999", lw=0.8, ls="--")
    ax.text(0.98, 0.98, "look real +\ncould co-locate\n→ risky", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#b0413a")
    ax.text(0.98, 0.02, "look real +\ncan't co-locate\n→ strong non-pair", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#b07d18")
    ax.text(0.02, 0.5, "don't look real\n→ easy", transform=ax.transAxes,
            ha="left", va="center", fontsize=8.5, color="#666")
    ax.set_xlabel("looks like a real interaction   (network proximity) →")
    ax.set_ylabel("biology says they can interact   (shared compartments) →")
    ax.set_title("Two lenses: does it look real, and can biology allow it?")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    if total < 20:
        ax.text(0.5, 0.5, "needs compartment annotations\n(run scripts/build_huri_annotations.py"
                "\nand view --dataset huri)", transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color="#999")
    fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_confidence_hardness(records, out_path: str | Path) -> Path:
    """The regime map on negaverse's own axes (Lucy's framing): every emitted
    negative by confidence (x) and hardness (y). Eval negatives sit safe/low-
    hardness; train negatives are hard/near-boundary; the `suspected_false_negative`
    (risky) tail is the low-confidence corner — the pairs that look positive-like."""
    import matplotlib.pyplot as plt
    groups = {
        "benchmark set (confident)": ([], "#457b9d", 0.5),
        "training set (challenging)": ([], "#e9c46a", 0.6),
        "risky — may really interact": ([], "#e63946", 0.9),
    }
    for r in records:
        key = ("risky — may really interact" if _is_risky(r)
               else "training set (challenging)" if r.mode == "train"
               else "benchmark set (confident)")
        groups[key][0].append((r.confidence, r.hardness))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, (pts, col, a) in groups.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=16, alpha=a, color=col, label=f"{name} (n={len(pts)})")
    ax.set_xlabel("how sure we are they DON'T interact  →")
    ax.set_ylabel("how much it still looks like a real interaction  →")
    ax.set_title("How confident, and how real-looking, each non-pair is")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_flag_breakdown(records, out_path: str | Path) -> Path:
    """How many emitted negatives carry each provenance flag (why a pair is what
    it is): different_compartment, near_boundary, suspected_false_negative, …"""
    import matplotlib.pyplot as plt
    from collections import Counter
    c = Counter(f for r in records for f in r.flags)
    if not c:
        c = Counter({"(no flags)": len(records)})
    labels, vals = zip(*c.most_common())         # raw provenance flag ids (audit panel)
    fig, ax = plt.subplots(figsize=(8.5, max(2.4, 0.55 * len(labels) + 1)))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color="#2a9d8f", height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v}", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontfamily="monospace", fontsize=9)
    ax.set_xlabel("number of emitted pairs")
    ax.set_title("Provenance flags per emitted pair")
    ax.margins(x=0.15); fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_manifold(graph: TypedInteractionGraph, records, out_path: str | Path,
                  seed: int = 0, n_ref: int = 500,
                  gold_negatives: list[tuple[str, str]] | None = None) -> Path:
    """Lucy's 4-regime manifold: PCA of pairs in topology-feature space —
    positives (the manifold), random negatives (far), hard negatives (close but
    distinguishable), and risky negatives (inside the positive-like cloud).

    gold_negatives: see plot_quadrant — a dataset's own labelled negatives,
    plotted instead of freshly-generated random pairs when given."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from ..bench.benchmark import _features
    adj = {n: set(graph.g.neighbors(n)) for n in graph.g.nodes()}
    edges = [tuple(e) for e in graph.g.edges()]
    edges = _stratified_sample(graph, edges, n_ref, seed)
    hard = [(r.u, r.v) for r in records if r.mode == "train" and not _is_risky(r)]
    risky = [(r.u, r.v) for r in records if _is_risky(r)]
    if gold_negatives:
        rand, rand_label = _stratified_sample(graph, gold_negatives, n_ref, seed), "dataset gold negatives"
    else:
        rand, rand_label = _random_nonedges(graph, n_ref, seed), "random non-pairs"
    cats = [("real interactions", edges, "#2a9d8f", 0.5),
            (rand_label, rand, "#adb5bd", 0.5),
            ("our chosen non-pairs", hard, "#e9c46a", 0.7),
            ("risky — may interact", risky, "#e63946", 0.9)]
    cats = [(n, p, c, a) for n, p, c, a in cats if p]
    allp = [pr for _, p, _, _ in cats for pr in p]
    # network measures are non-negative counts with hub-driven heavy tails
    # (degree, preferential attachment) — log-compress before standardizing so a
    # few hubs don't dominate the layout.
    X = np.log1p(_features(adj, allp))
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    xy = PCA(2, random_state=seed).fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(7.6, 6))
    i = 0
    for name, p, col, a in cats:
        j = i + len(p)
        ax.scatter(xy[i:j, 0], xy[i:j, 1], s=14, alpha=a, color=col,
                   label=f"{name} (n={len(p)})")
        i = j
    ax.set_xlabel("network-feature map — axis 1")
    ax.set_ylabel("network-feature map — axis 2")
    ax.set_title("Map of protein pairs (many network measures squeezed into 2D)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def render_all(graph: TypedInteractionGraph, records, out_dir: str | Path,
               stats: dict | None = None, seed: int = 0, n_ref: int = 500,
               x_axis: "tuple | None" = None,
               gold_negatives: list[tuple[str, str]] | None = None):
    """Render every demo panel from a pipeline run's records (+ stats).

    x_axis (optional) overrides the interactive map's default topology x-axis with
    another looks-real lens — see compute_traces (used for DRYAD's sequence axis).

    gold_negatives (optional): a dataset's own labelled negative benchmark
    (DRYAD/UPNA) — plotted in place of freshly-generated random pairs across
    every panel below, since a real external negative set is a more
    meaningful comparison than a synthetic one when a dataset ships one."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = [tuple(e) for e in graph.g.edges()]
    edges = _stratified_sample(graph, edges, n_ref, seed)
    hard = [(r.u, r.v) for r in records]
    n_neg = min(n_ref, max(len(hard), 1))
    if gold_negatives:
        random_neg, random_label = _stratified_sample(graph, gold_negatives, n_neg, seed), "dataset gold negatives"
    else:
        random_neg, random_label = _random_nonedges(graph, n_neg, seed), None

    written = [plot_separability(graph, edges, random_neg, hard, out_dir / "separability.png",
                                 random_label=random_label)]
    if records:
        written.append(plot_quadrant(graph, records, out_dir / "quadrant.png", seed, n_ref,
                                     gold_negatives=gold_negatives))
        written.append(plot_manifold(graph, records, out_dir / "manifold.png", seed, n_ref,
                                     gold_negatives=gold_negatives))
        written.append(plot_confidence_hardness(records, out_dir / "confidence_hardness.png"))
        written.append(plot_flag_breakdown(records, out_dir / "flag_breakdown.png"))
    if stats:
        written.append(plot_funnel(stats, out_dir / "funnel.png"))
    if records:
        try:
            import json as _json
            from .interactive import compute_traces
            (out_dir / "interactive3d.json").write_text(
                _json.dumps(compute_traces(graph, records, seed, n_ref, x_axis=x_axis,
                                          gold_negatives=gold_negatives)))
            written.append(out_dir / "interactive3d.json")
        except Exception:
            pass                          # 3D panel is optional
    return written
