"""Manifold-surprisal filters — resemblance to the frozen positive manifold.

Each protein is placed in some embedding space; a candidate pair is scored by how
much it resembles the crowd of *real* interactions. Deep inside the crowd => looks
like a true edge => a suspected false negative (risky); far outside => safe. Two
embedding spaces, two filters over one shared mechanism:

  * ManifoldSurprisalFilter — a truncated-SVD embedding of the interaction graph
    ("who it interacts with"). The GLOBAL-graph companion to the LOCAL
    TopologyFilter; correlated with it (~0.64) but not identical, so it earns its
    keep as a flag / disagreement signal, not a selection driver.
  * SequenceManifoldFilter — a per-protein SEQUENCE embedding (e.g. ESM2). The
    genuinely INDEPENDENT axis (correlation ~0.2 with the graph views), so it is
    the highest-value fusion partner and the natural fit for sequence-rich PLI
    (ESM2 for protein, MolFormer for ligand). Needs embeddings supplied; abstains
    for any node without one. See docs/IG-FEATURES.md §3b/§3c.

Both are `default = False` (opt-in): registered and buildable by name, but not in
the default pipeline. Shared guardrails, like the other filters: calibrate the
resemblance scale at fit against held-out real edges; abstain when a node has no
embedding; build the manifold only from the graph fitted on (no leakage).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ..graph import TypedInteractionGraph
from ..ig.surprisal import background_similarity, normalize_rows
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register

_EMB_DIM = 32           # truncated-SVD embedding dimension (spectral)
_TOPK = 10              # top-k nearest real pairs averaged for resemblance
_SEQ_TOPK = 25          # sequence embeddings are denser; a wider k is steadier
_BG_SAMPLE = 2000       # real edges frozen as the positive-manifold background
_CAL_SAMPLE = 2000      # held-out real edges used to calibrate scale / FN threshold
_FN_QUANTILE = 0.5      # a pair as positive-like as a typical real edge => flag


class _ManifoldSurprisalBase(Filter):
    """Shared machinery: build a frozen positive manifold from real edges, then
    score a pair by top-k-mean cosine resemblance to it. Subclasses supply the
    node embeddings (`_node_embeddings`) and the pair operator (`_OP`)."""

    stage = Stage.GRADED
    default = False
    _OP = "hadamard"       # how two node vectors combine into a pair vector

    def __init__(self, topk: int = _TOPK, seed: int = 0) -> None:
        self._topk = topk
        self._seed = seed
        self._emb: dict[str, np.ndarray] = {}     # node -> embedding (usable nodes only)
        self._bg: np.ndarray | None = None        # normalised positive-manifold pair reps
        self._scale: float = 1.0                  # saturating scale = median edge resemblance
        self._flag_thresh: float = 1.0            # resemblance ≥ this => suspected FN

    # subclasses implement this
    def _node_embeddings(self, graph: TypedInteractionGraph) -> dict[str, np.ndarray]:
        raise NotImplementedError

    # --- fit -------------------------------------------------------------
    def fit(self, graph: TypedInteractionGraph) -> None:
        self._emb = self._node_embeddings(graph)
        edges = [(u, v) for u, v in graph.g.edges()
                 if u in self._emb and v in self._emb]
        if not edges:
            self._bg = None
            return
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(len(edges))
        bg_ids = perm[:min(len(edges), _BG_SAMPLE)]
        self._bg = normalize_rows(self._pair_reps([edges[i] for i in bg_ids]))
        # calibrate on edges NOT in the background (avoid self-resemblance); on a
        # tiny graph, fall back to all edges.
        cal_ids = perm[len(bg_ids):len(bg_ids) + _CAL_SAMPLE]
        if len(cal_ids) < 50:
            cal_ids = perm[:min(len(edges), _CAL_SAMPLE)]
        res = self._resemblance([edges[i] for i in cal_ids])
        res = res[res > 0]
        if res.size:
            self._scale = float(np.median(res))
            self._flag_thresh = float(np.quantile(res, _FN_QUANTILE))

    # --- pair geometry ---------------------------------------------------
    def _pair_rep(self, u: str, v: str) -> np.ndarray:
        eu, ev = self._emb[u], self._emb[v]
        if self._OP == "hadamard":
            return eu * ev
        if self._OP == "avg":
            return 0.5 * (eu + ev)
        if self._OP == "concat":                  # order-invariant (min,max) concat
            return np.concatenate([np.minimum(eu, ev), np.maximum(eu, ev)])
        raise ValueError(f"unknown operator {self._OP!r}")

    def _pair_reps(self, pairs) -> np.ndarray:
        return np.asarray([self._pair_rep(u, v) for u, v in pairs], dtype=float)

    def _resemblance(self, pairs) -> np.ndarray:
        return background_similarity(normalize_rows(self._pair_reps(pairs)),
                                     self._bg, k=self._topk)

    # --- score -----------------------------------------------------------
    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        if self._bg is None or u not in self._emb or v not in self._emb:
            return StreamScore(self.name, value=None,
                               evidence={"status": "no_embedding"})
        r = float(self._resemblance([(u, v)])[0])
        rr = max(r, 0.0)
        risk = rr / (rr + self._scale) if self._scale > 0 else rr
        value = round(1.0 - risk, 4)
        flags = ["suspected_false_negative"] if r >= self._flag_thresh else []
        return StreamScore(
            self.name, value=value, flags=flags,
            evidence={"resemblance": round(r, 4),
                      "edge_median": round(self._scale, 4),
                      "risk": round(risk, 4),
                      # peakedness for entropy-weighted fusion (ig/entropy_fusion)
                      "confidence": round(abs(2 * value - 1), 4)},
        )


@register
class ManifoldSurprisalFilter(_ManifoldSurprisalBase):
    """Global-graph manifold: truncated-SVD embedding of the interaction graph."""

    name = "manifold"
    modalities = frozenset({"ppi"})
    _OP = "hadamard"                              # Hadamard wins for graph embeddings

    def __init__(self, dim: int = _EMB_DIM, topk: int = _TOPK, seed: int = 0) -> None:
        super().__init__(topk=topk, seed=seed)
        self._dim = dim

    def _node_embeddings(self, graph: TypedInteractionGraph) -> dict[str, np.ndarray]:
        nodes = list(graph.g.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        rows, cols = [], []
        for a, b in graph.g.edges():
            i, j = idx[a], idx[b]
            rows += [i, j]
            cols += [j, i]
        if not rows:
            return {}
        n = len(nodes)
        k = max(1, min(self._dim, n - 2))
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).asfptype()
        rng = np.random.default_rng(self._seed)
        v0 = rng.standard_normal(min(A.shape))
        U, S, _ = svds(A, k=k, v0=v0)
        M = U * S                                 # scale components by singular value
        return {nodes[i]: M[i] for i in range(n) if float(np.linalg.norm(M[i])) > 0.0}


@register
class SequenceManifoldFilter(_ManifoldSurprisalBase):
    """Sequence manifold: a per-protein embedding (e.g. ESM2), the independent
    axis. Supply `embeddings` (node id -> vector) or a `.npz` `path` with `ids`
    and `emb` arrays; abstains for any node without an embedding."""

    name = "sequence_manifold"
    modalities = frozenset({"ppi", "pli"})
    _OP = "concat"                               # concat/avg beat Hadamard for ESM2

    def __init__(self, embeddings: "dict[str, np.ndarray] | None" = None,
                 path: str | None = None, topk: int = _SEQ_TOPK, seed: int = 0) -> None:
        super().__init__(topk=topk, seed=seed)
        self._embeddings = embeddings
        self._path = path

    def _node_embeddings(self, graph: TypedInteractionGraph) -> dict[str, np.ndarray]:
        raw = self._embeddings
        if raw is None and self._path:
            from ..io.embeddings import load_embeddings_npz
            raw = load_embeddings_npz(self._path)
        raw = raw or {}
        return {n: np.asarray(raw[n], dtype=float)
                for n in graph.g.nodes() if n in raw}
