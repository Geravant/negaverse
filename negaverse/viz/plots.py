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
                 color=_C[name], label=f"{name} (n={len(pairs)})")
    ax1.set_xlabel(f"common neighbours (clipped at {clip})")
    ax1.set_ylabel("density")
    ax1.set_title("Common-neighbour overlap")
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
                color=_C[name], label=name)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("shortest-path length in the PPI graph")
    ax2.set_ylabel("proportion of pairs")
    ax2.set_title("Graph distance to the interaction network")
    ax2.legend()

    fig.suptitle("Separability: negaverse hard negatives vs random (positives = reference)",
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
        ("candidates", cand),
        ("survived VETO", cand - vetoed),
        ("scored (GRADED)", scored),
        ("reviewed (GATED)", gated),
        ("emitted", emitted),
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
    ax.set_xlabel("pairs")
    ax.set_title("Hourglass funnel — pairs kept per stage")
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


def render_all(graph: TypedInteractionGraph, records, out_dir: str | Path,
               stats: dict | None = None, seed: int = 0, n_ref: int = 500):
    """Render the demo panels from a pipeline run's records (+ stats)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    edges = [tuple(e) for e in graph.g.edges()]
    if len(edges) > n_ref:
        idx = rng.choice(len(edges), size=n_ref, replace=False)
        edges = [edges[i] for i in idx]
    hard = [(r.u, r.v) for r in records]
    random_neg = _random_nonedges(graph, min(n_ref, max(len(hard), 1)), seed)

    written = [plot_separability(graph, edges, random_neg, hard,
                                 out_dir / "separability.png")]
    if stats:
        written.append(plot_funnel(stats, out_dir / "funnel.png"))
    return written
