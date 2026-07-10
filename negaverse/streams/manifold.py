"""Manifold-surprisal filter — global-graph resemblance to the positive manifold.

The companion to the (local) TopologyFilter. Topology reads *local* structure
(shared neighbours, length-3 paths); this reads the *global* embedding geometry:
each protein is placed by a truncated-SVD embedding of the interaction graph
("who it interacts with"), and a candidate pair is scored by how much it
resembles the frozen crowd of real interactions.

  * Deep inside that crowd  => looks like a true edge => *risky* negative (a
    suspected false negative). This is the filter's headline job — the
    `suspected_false_negative` flag that keeps such pairs out of the clean eval
    set.
  * Far outside the crowd    => looks nothing like an interaction => *safe*.

The two graph views are correlated but not identical (a second witness on the
same graph), so this is introduced as a *flag / confidence* signal, **not** as a
hard-negative selection driver — using it to select and then grading on the same
embedding features would be circular (see docs/IG-FEATURES.md, docs/BENCHMARK-
FINDINGS.md). It is therefore `default = False`: registered and buildable, opted
into by name until the feature-independent downstream check clears it.

Guardrails, matching the other filters:
  * resemblance is squashed through the saturating map `r/(r+scale)` whose
    `scale` is calibrated at fit to the median resemblance of *held-out real
    edges*, so `value`/`risk` are comparable across graphs of different density
    (cf. TopologyFilter's L3/RA calibration);
  * abstains (value=None) when a node has no usable embedding (isolated / tiny
    component) — geometry with no support must not guess;
  * builds the manifold only from the graph it is fitted on (no leakage).
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

_EMB_DIM = 32           # truncated-SVD embedding dimension
_TOPK = 10              # top-k nearest real pairs averaged for resemblance
_BG_SAMPLE = 2000       # real edges frozen as the positive-manifold background
_CAL_SAMPLE = 2000      # held-out real edges used to calibrate scale / FN threshold
_FN_QUANTILE = 0.5      # a pair as positive-like as a typical real edge => flag


@register
class ManifoldSurprisalFilter(Filter):
    name = "manifold"
    stage = Stage.GRADED
    modalities = frozenset({"ppi"})
    default = False       # opt-in: heavier (SVD) and flag-first pending validation

    def __init__(self, dim: int = _EMB_DIM, topk: int = _TOPK, seed: int = 0) -> None:
        self._dim = dim
        self._topk = topk
        self._seed = seed
        self._idx: dict[str, int] = {}
        self._emb: np.ndarray | None = None
        self._bg: np.ndarray | None = None       # normalised positive-manifold pair reps
        self._scale: float = 1.0                  # saturating scale = median edge resemblance
        self._flag_thresh: float = 1.0            # resemblance ≥ this => suspected false negative

    # --- fit -------------------------------------------------------------
    def fit(self, graph: TypedInteractionGraph) -> None:
        g = graph.g
        nodes = list(g.nodes())
        self._idx = {n: i for i, n in enumerate(nodes)}
        self._emb = self._spectral_embeddings(g, nodes)
        edges = list(g.edges())
        if self._emb is None or not edges:
            return
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(len(edges))
        bg_ids = perm[:min(len(edges), _BG_SAMPLE)]
        self._bg = normalize_rows(self._pair_reps([edges[i] for i in bg_ids]))
        # calibrate on edges NOT in the background sample (avoid self-resemblance);
        # fall back to all edges on tiny graphs.
        cal_ids = perm[len(bg_ids):len(bg_ids) + _CAL_SAMPLE]
        if len(cal_ids) < 50:
            cal_ids = perm[:min(len(edges), _CAL_SAMPLE)]
        self._calibrate([edges[i] for i in cal_ids])

    def _calibrate(self, edges: list[tuple[str, str]]) -> None:
        res = self._resemblance(edges)
        res = res[res > 0]
        if res.size:
            self._scale = float(np.median(res))
            self._flag_thresh = float(np.quantile(res, _FN_QUANTILE))

    def _spectral_embeddings(self, g, nodes: list[str]) -> "np.ndarray | None":
        """Truncated-SVD node embeddings of the (symmetric) adjacency, scaled by
        singular value — the standard node2vec-style graph embedding."""
        rows, cols = [], []
        for a, b in g.edges():
            i, j = self._idx[a], self._idx[b]
            rows += [i, j]
            cols += [j, i]
        if not rows:
            return None
        n = len(nodes)
        k = max(1, min(self._dim, n - 2))
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).asfptype()
        rng = np.random.default_rng(self._seed)
        v0 = rng.standard_normal(min(A.shape))
        U, S, _ = svds(A, k=k, v0=v0)
        return U * S

    # --- pair geometry ---------------------------------------------------
    def _pair_reps(self, pairs) -> np.ndarray:
        d = self._emb.shape[1]
        zero = np.zeros(d)
        out = []
        for u, v in pairs:
            iu, iv = self._idx.get(u), self._idx.get(v)
            out.append(self._emb[iu] * self._emb[iv]      # Hadamard (node2vec link feature)
                       if iu is not None and iv is not None else zero)
        return np.asarray(out, dtype=float)

    def _resemblance(self, pairs) -> np.ndarray:
        reps = normalize_rows(self._pair_reps(pairs))
        return background_similarity(reps, self._bg, k=self._topk)

    def _has_embedding(self, n: str) -> bool:
        i = self._idx.get(n)
        return (self._emb is not None and i is not None
                and float(np.linalg.norm(self._emb[i])) > 0.0)

    # --- score -----------------------------------------------------------
    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        if self._bg is None or not self._has_embedding(u) or not self._has_embedding(v):
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
