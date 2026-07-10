"""How well does negaverse's TOPOLOGY signal separate real PPIs from three matched
negative sets on the UPNA-PPI data (github.com/alxndgb/UPNA-PPI)?

UPNA-PPI is itself a topology-driven negative-sampling paper (contrastive-L3), so
this is a head-to-head: does *our* topology risk (L3 + resource-allocation +
config-model) score positives above their negatives? Three matched negative sets,
one protein universe (the 5,037 proteins that appear in their topological negatives):

  * PPNI        — UPNA-PPI's ML-derived non-interactions
  * topological — UPNA-PPI's contrastive-L3 negatives (their headline method)
  * random      — uniform non-edges (our baseline)

Separation = AUROC of the topology risk, positives (1) vs each negative set (0).
AUROC ~0.5 => topology can't tell them apart (hard negatives); ~1.0 => trivial.

Scale note: the ComPPlete positive interactome is huge and dense (17,974 proteins,
4.58M edges, mean degree ~510). Our production TopologyFilter uses the same L3+RA
formulas but a set-based inner loop, which is fine for sparse experimental graphs
(HuRI ~12) yet intractable here. So we compute the **identical** risk via
scipy.sparse — exactly the vectorised form the IMPLEMENTATION-PLAN specifies:
degree-normalised L3 is `A · M · A` with `M = D^{-1/2} A D^{-1/2}`, RA is the
common-neighbour indicator dotted with 1/deg. Same signal, done at scale.

    PYTHONPATH=. python scripts/upna_topology_separation.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

DATA = Path("local-docs/upna-ppi")
PPI_PARTS = sorted(DATA.glob("PPI_part_*.csv"))
PPNI_FILE = DATA / "PPNI_part_2.csv"
TPPNI_FILES = sorted(DATA.glob("TPPNI_*.csv"))

SEED = 0
N_PER_CLASS = 3000
TEST_FRAC = 0.4            # held-out fraction of the in-universe positives
L3_W, RA_W = 0.7, 0.3     # same weights as TopologyFilter
FLOOR = 0.02              # risk floor for no-overlap pairs (1 - 0.98)


def _pairs(df):
    return list(zip(df["SymbolA"].astype(str), df["SymbolB"].astype(str)))


def _load_topological():
    pairs = []
    for f in TPPNI_FILES:
        pairs += [(a, b) for a, b in _pairs(pd.read_csv(f, usecols=["SymbolA", "SymbolB"]))
                  if a != b]
    return pairs, {x for p in pairs for x in p}


def _stream_edges(files):
    edges, seen = [], set()
    for f in files:
        for ch in pd.read_csv(f, usecols=["SymbolA", "SymbolB"], chunksize=300_000):
            for a, b in zip(ch["SymbolA"].astype(str), ch["SymbolB"].astype(str)):
                if a == b:
                    continue
                k = (a, b) if a < b else (b, a)
                if k in seen:
                    continue
                seen.add(k)
                edges.append(k)
    return edges, seen


def _sample(pairs, n, rng):
    if len(pairs) <= n:
        return list(pairs)
    return [pairs[i] for i in rng.choice(len(pairs), size=n, replace=False)]


class SparseTopology:
    """Vectorised equivalent of TopologyFilter's L3+RA+config risk."""

    def __init__(self, edges, nodes):
        self.idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        r = [self.idx[a] for a, b in edges] + [self.idx[b] for a, b in edges]
        c = [self.idx[b] for a, b in edges] + [self.idx[a] for a, b in edges]
        self.A = sp.csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)), shape=(n, n))
        self.A.data[:] = 1.0
        self.deg = np.asarray(self.A.sum(1)).ravel()
        self.two_m = float(self.deg.sum()) or 1.0
        inv_sqrt = np.divide(1.0, np.sqrt(self.deg), where=self.deg > 0,
                             out=np.zeros_like(self.deg))
        D = sp.diags(inv_sqrt)
        self.M = (D @ self.A @ D).tocsr()               # normalized adjacency
        self.inv_deg = np.divide(1.0, self.deg, where=self.deg > 0,
                                 out=np.zeros_like(self.deg))
        self.l3_scale = self.ra_scale = 1.0

    def _raw(self, pairs):
        """Return (l3, ra, cn) arrays for pairs (skips unknown nodes)."""
        rows = [self.idx.get(u, -1) for u, _ in pairs]
        cols = [self.idx.get(v, -1) for _, v in pairs]
        l3 = np.zeros(len(pairs)); ra = np.zeros(len(pairs)); cn = np.zeros(len(pairs))
        for k, (i, j) in enumerate(zip(rows, cols)):
            if i < 0 or j < 0:
                continue
            ai, aj = self.A[i], self.A[j]
            shared = ai.multiply(aj).tocsr()            # common-neighbour indicator
            cn[k] = shared.nnz
            if shared.nnz:
                ra[k] = float(self.inv_deg[shared.indices].sum())
            l3[k] = float((ai @ self.M).multiply(aj).sum())
        return l3, ra, cn

    def calibrate(self, pos_edges, rng):
        s = _sample(pos_edges, min(2000, len(pos_edges)), rng)
        l3, ra, _ = self._raw(s)
        self.l3_scale = float(np.median(l3[l3 > 0])) if (l3 > 0).any() else 1.0
        self.ra_scale = float(np.median(ra[ra > 0])) if (ra > 0).any() else 1.0

    def risk(self, pairs):
        l3, ra, cn = self._raw(pairs)
        out = np.full(len(pairs), FLOOR)                # no-overlap -> floor
        hit = (cn > 0) | (l3 > 0)
        l3n = l3 / (l3 + self.l3_scale)
        ran = np.divide(ra, ra + self.ra_scale, where=ra > 0, out=np.zeros_like(ra))
        out[hit] = (L3_W * l3n + RA_W * ran)[hit]
        return out


def main():
    rng = np.random.default_rng(SEED)
    print("Loading topological negatives + universe ...")
    topo_neg, universe = _load_topological()
    print(f"  topological negs: {len(topo_neg):,}   universe: {len(universe):,} proteins")

    print("Streaming full PPI positive graph (17,974 proteins, ~4.58M edges) ...")
    edges, edge_set = _stream_edges(PPI_PARTS)
    nodes = sorted({x for e in edges for x in e} | universe)
    print(f"  graph: {len(nodes):,} nodes, {len(edges):,} edges")

    pos_uni = [e for e in edges if e[0] in universe and e[1] in universe]
    rng.shuffle(pos_uni)
    n_test = int(len(pos_uni) * TEST_FRAC)
    test_pos = pos_uni[:n_test]
    test_set = set(test_pos)
    train_edges = [e for e in edges if e not in test_set]     # leakage-safe graph
    print(f"  in-universe positives: {len(pos_uni):,}  (held out {len(test_pos):,})")

    print("Building sparse graph + calibrating (same L3+RA+config risk) ...")
    st = SparseTopology(train_edges, nodes)
    st.calibrate([e for e in train_edges if e[0] in universe and e[1] in universe], rng)

    # random non-edges within the universe
    uni = sorted(universe)
    rand = []
    seenr = set()
    while len(rand) < N_PER_CLASS:
        a, b = uni[rng.integers(len(uni))], uni[rng.integers(len(uni))]
        k = (a, b) if a < b else (b, a)
        if a != b and k not in edge_set and k not in seenr:
            seenr.add(k); rand.append((a, b))

    neg_sets = {"PPNI": _stream_edges([PPNI_FILE])[0],
                "topological": [p for p in topo_neg if p[0] in universe and p[1] in universe],
                "random": rand}

    rp = st.risk(_sample(test_pos, N_PER_CLASS, rng))
    print(f"\nheld-out positives: mean risk {rp.mean():.3f}\n")
    print("=== topology separation: AUROC(positive vs negative) ===")
    print("  (0.50 = indistinguishable / hard; 1.0 = trivially separable)\n")
    risks = {}
    for name, negs in neg_sets.items():
        rn = st.risk(_sample(negs, N_PER_CLASS, rng))
        risks[name] = rn
        y = np.r_[np.ones(len(rp)), np.zeros(len(rn))]
        auroc = roc_auc_score(y, np.r_[rp, rn])
        frac = float((rn > FLOOR).mean())
        print(f"  {name:12} AUROC={auroc:.3f}   mean risk pos={rp.mean():.3f} "
              f"neg={rn.mean():.3f}   neg with overlap={frac:.0%}")
    _plot(rp, risks)


def _plot(rp, risks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    out = Path("out"); out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    bins = np.linspace(0, 1, 31)
    for ax, (name, rn) in zip(axes, risks.items()):
        au = roc_auc_score(np.r_[np.ones(len(rp)), np.zeros(len(rn))], np.r_[rp, rn])
        ax.hist(rp, bins=bins, density=True, histtype="step", lw=2, color="#2a9d8f", label="positive")
        ax.hist(rn, bins=bins, density=True, histtype="step", lw=2, color="#e76f51", label=f"{name} neg")
        ax.set_title(f"{name}: AUROC={au:.3f}"); ax.set_xlabel("topology risk (L3+RA+config)")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle("negaverse topology separation on UPNA-PPI (positives vs 3 matched negatives)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "upna_topology_separation.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
