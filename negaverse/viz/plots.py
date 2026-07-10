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
                      out_path: str | Path) -> Path:
    """Show negaverse's hard negatives sit *between* positives and random
    negatives on structural axes — i.e. they are the topologically positive-like
    (harder) negatives the tool is meant to select."""
    g = graph.g
    sets = {"positive": positives, "random": random_neg, "hard": hard_neg}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # panel 1: common-neighbour count (clipped for readability)
    clip = 12
    for name, pairs in sets.items():
        if not pairs:
            continue
        cn = np.clip(_common_neighbors(g, pairs), 0, clip)
        bins = np.arange(0, clip + 2) - 0.5
        ax1.hist(cn, bins=bins, density=True, histtype="step", linewidth=2,
                 color=_C[name], label=f"{_NAME[name]} (n={len(pairs)})")
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
                color=_C[name], label=_NAME[name])
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("steps apart in the interaction network")
    ax2.set_ylabel("share of pairs")
    ax2.set_title("How far apart the two proteins are")
    ax2.legend()

    fig.suptitle("Our chosen non-pairs look more like real interactions than random ones do",
                 fontsize=12)
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
    rng = np.random.default_rng(seed)
    nodes = list(graph.g.nodes())
    N = len(nodes)
    out, seen = [], set()
    tries, cap = 0, n * 60 + 1000
    while len(out) < n and tries < cap:
        tries += 1
        a, b = nodes[rng.integers(N)], nodes[rng.integers(N)]
        if a == b or graph.g.has_edge(a, b):
            continue
        k = frozenset((a, b))
        if k in seen:
            continue
        seen.add(k)
        out.append((a, b))
    return out


def _is_risky(r) -> bool:
    return "suspected_false_negative" in r.flags


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
        c = Counter({"(no notes)": len(records)})
    labels, vals = zip(*c.most_common())
    labels = [_flag_label(l) for l in labels]
    fig, ax = plt.subplots(figsize=(8.5, max(2.4, 0.55 * len(labels) + 1)))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color="#2a9d8f", height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v}", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("number of pairs"); ax.set_title("Why each pair was labelled the way it was")
    ax.margins(x=0.15); fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_manifold(graph: TypedInteractionGraph, records, out_path: str | Path,
                  seed: int = 0, n_ref: int = 500) -> Path:
    """Lucy's 4-regime manifold: PCA of pairs in topology-feature space —
    positives (the manifold), random negatives (far), hard negatives (close but
    distinguishable), and risky negatives (inside the positive-like cloud)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from ..bench.benchmark import _features
    adj = {n: set(graph.g.neighbors(n)) for n in graph.g.nodes()}
    rng = np.random.default_rng(seed)
    edges = [tuple(e) for e in graph.g.edges()]
    if len(edges) > n_ref:
        edges = [edges[i] for i in rng.choice(len(edges), n_ref, replace=False)]
    hard = [(r.u, r.v) for r in records if r.mode == "train" and not _is_risky(r)]
    risky = [(r.u, r.v) for r in records if _is_risky(r)]
    rand = _random_nonedges(graph, n_ref, seed)
    cats = [("real interactions", edges, "#2a9d8f", 0.5),
            ("random non-pairs", rand, "#adb5bd", 0.5),
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
    ax.set_xlabel("each dot is a protein pair — similar pairs sit near each other")
    ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Map of protein pairs")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def render_all(graph: TypedInteractionGraph, records, out_dir: str | Path,
               stats: dict | None = None, seed: int = 0, n_ref: int = 500):
    """Render every demo panel from a pipeline run's records (+ stats)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    edges = [tuple(e) for e in graph.g.edges()]
    if len(edges) > n_ref:
        idx = rng.choice(len(edges), size=n_ref, replace=False)
        edges = [edges[i] for i in idx]
    hard = [(r.u, r.v) for r in records]
    random_neg = _random_nonedges(graph, min(n_ref, max(len(hard), 1)), seed)

    written = [plot_separability(graph, edges, random_neg, hard, out_dir / "separability.png")]
    if records:
        written.append(plot_manifold(graph, records, out_dir / "manifold.png", seed, n_ref))
        written.append(plot_confidence_hardness(records, out_dir / "confidence_hardness.png"))
        written.append(plot_flag_breakdown(records, out_dir / "flag_breakdown.png"))
    if stats:
        written.append(plot_funnel(stats, out_dir / "funnel.png"))
    return written
