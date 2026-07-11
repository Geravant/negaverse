"""Paper-style evaluation of negaverse negatives — the UPNA-PPI / TPPNI protocol
(Chatterjee & Ravandi et al., Bioinformatics 2025, btaf148), applied to negaverse.

negaverse's own bench is TRANSDUCTIVE (random edge split) + SPECTRAL features +
AUROC. The paper argues that's the wrong instrument: hard-negative value shows up
INDUCTIVELY (novel proteins) under LOCAL RANKING metrics, not transductively under
global AUROC. This bench reproduces their protocol so we can see whether negaverse's
rules earn their keep when measured the paper's way:

  * INDUCTIVE split (GraIL): partition PROTEINS into disjoint train/test groups, so
    test proteins are never seen in training. test positives/negatives involve only
    test-group proteins.
  * SEQUENCE features (ESM2, node-intrinsic) so unseen proteins can be featurized —
    the paper uses ProtVec; DRYAD ships ESM2. Topology is used only to SELECT training
    negatives, never as a feature (so it generalizes to novel proteins).
  * LOCAL RANKING metrics: PPIHits@TopK (precision of positives in the top-K scored)
    and PPNIHits@BottomK (precision of negatives in the bottom-K) — plus AUROC/AUPRC
    for reference. Paper Table 1: random gets decent AUROC but collapses at the tails.
  * Strategy comparison (paper's Table 1 shape): random vs topology vs stacked(+rules),
    so the rules table is evaluated head-to-head with the paper's own baselines.

Only DRYAD is supported: it ships ESM2 embeddings (node-intrinsic features are
required for an inductive test; HuRI has none). Restricted to the embedded subgraph.

    PYTHONPATH=. python3 scripts/bench_paper_style.py [--seeds 0 1 2] [--split 0.5] [--k 100]
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_embeddings_npz
from negaverse.pipeline import PipelineConfig, run_pipeline
from negaverse.rule_engine import load_rules
from negaverse.streams import build_filters
from negaverse.streams.rules import RuleGradedFilter

EMB_PATH = "local-docs/dryad-ppi/esm2_t6_emb.npz"
DRYAD_TSV = "local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv"


def _load_dryad_embedded():
    pos, neg = [], []
    with open(DRYAD_TSV) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    emb = load_embeddings_npz(EMB_PATH)
    E = set(emb)
    pos = [(u, v) for u, v in pos if u in E and v in E]
    neg = [(u, v) for u, v in neg if u in E and v in E]
    return pos, neg, emb


def _hadamard(emb, pairs):
    dim = len(next(iter(emb.values())))
    zero = np.zeros(dim)
    return np.asarray([emb.get(u, zero) * emb.get(v, zero) for u, v in pairs], dtype=float)


def _topology_stacked_negatives(train_pos, n, seed, rule_subset):
    """Hard negatives among TRAIN-node pairs, via the pipeline on the train
    subgraph. rule_subset=None => structured+topology only (topology strategy);
    rule_subset=all => stacked (topology + graded rules)."""
    nodes = {p for e in train_pos for p in e}
    tg = TypedInteractionGraph.from_edges(
        list(train_pos), {n_: "protein" for n_ in nodes},
        admissible_types=[("protein", "protein")], name="ind-train")
    filters = build_filters("ppi", ["known_positive_veto", "structured", "topology"])
    if rule_subset:
        filters = filters + [RuleGradedFilter(rules=list(rule_subset))]
    cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=max(4 * n, n), max_pool=40000, seed=seed)
    res = run_pipeline(tg, cfg, filters=filters)
    hard = [r for r in res.records if r.mode == "train"]
    hard.sort(key=lambda r: r.confidence, reverse=True)
    return [(r.u, r.v) for r in hard[:n]]


def _random_negatives(train_nodes, pos_set, n, rng):
    nodes = list(train_nodes)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 100 + 5000:
        tries += 1
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a != b and k not in pos_set and k not in seen:
            seen.add(k); out.append((a, b))
    return out


def _hits_at_k(y_true, scores, k, positive=True):
    """PPIHits@TopK (positive=True): precision of positives among the top-K scored.
    PPNIHits@BottomK (positive=False): precision of negatives among the bottom-K."""
    order = np.argsort(scores)                     # ascending
    idx = order[-k:] if positive else order[:k]
    target = 1 if positive else 0
    return float(np.mean(y_true[idx] == target))


def _mrr(y_true, scores):
    """Mean reciprocal rank of the positives (rank 1 = highest score)."""
    order = np.argsort(-scores)                    # descending
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    pr = ranks[y_true == 1]
    return float(np.mean(1.0 / pr)) if len(pr) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", type=float, default=0.5, help="fraction of proteins in the TRAIN group")
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    args = ap.parse_args()

    pos_all, neg_all, emb = _load_dryad_embedded()
    rules = [r for r in load_rules()
             if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]

    # Full battery: paper baselines (random, curated-experimental) + negaverse
    # (topology, stacked=all rules) + every rule added individually on top of topology.
    # spec: "random" | "curated" | None (topology) | list[Rule] (rule subset).
    strategies: dict = {"random": "random", "curated": "curated",
                        "topology": None, "stacked": rules}
    for r in rules:
        strategies[f"+{r.id}"] = [r]

    print("=" * 96)
    print("Paper-style (UPNA-PPI/TPPNI) inductive evaluation — DRYAD, ESM2 features — FULL BATTERY")
    print(f"pos(embedded)={len(pos_all)}  neg(embedded)={len(neg_all)}  "
          f"split={args.split:.0%} train proteins  seeds={args.seeds}")
    print(f"graded rules: {[r.id for r in rules]}")

    # firing coverage of each rule on the DRYAD (embedded) graph — so inert rules are visible
    allnodes = {p for e in pos_all + neg_all for p in e}
    _tg = TypedInteractionGraph.from_edges(
        list(pos_all), {n_: "protein" for n_ in allnodes},
        admissible_types=[("protein", "protein")], name="cov")
    _rng = np.random.default_rng(0); _ns = list(allnodes)
    _samp = [(_ns[_rng.integers(len(_ns))], _ns[_rng.integers(len(_ns))]) for _ in range(3000)]
    print("rule firing coverage on DRYAD (fraction of sampled pairs scored):")
    for r in rules:
        f = RuleGradedFilter(rules=[r]); f.fit(_tg)
        cov = sum(1 for u, v in _samp if u != v and f.score(_tg, u, v).value is not None) / len(_samp)
        print(f"  {r.id:<42} {cov:6.1%}")
    print("=" * 96)

    metrics = {s: {"auroc": [], "auprc": [], "mrr": [],
                   **{f"ppi@top{k}": [] for k in args.k},
                   **{f"ppni@bot{k}": [] for k in args.k}} for s in strategies}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        nodes = sorted({p for e in pos_all + neg_all for p in e})
        rng.shuffle(nodes)
        cut = int(len(nodes) * args.split)
        train_nodes, test_nodes = set(nodes[:cut]), set(nodes[cut:])

        train_pos = [(u, v) for u, v in pos_all if u in train_nodes and v in train_nodes]
        test_pos = [(u, v) for u, v in pos_all if u in test_nodes and v in test_nodes]
        test_neg = [(u, v) for u, v in neg_all if u in test_nodes and v in test_nodes]
        train_curated = [(u, v) for u, v in neg_all if u in train_nodes and v in train_nodes]
        if not train_pos or not test_pos or not test_neg:
            print(f"  seed {seed}: empty split, skipping"); continue
        pos_set = {frozenset(e) for e in pos_all}
        n_neg = len(train_pos)

        Xte = _hadamard(emb, list(test_pos) + list(test_neg))
        yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))].astype(int)

        for strat, spec in strategies.items():
            if spec == "random":
                neg = _random_negatives(train_nodes, pos_set, n_neg, rng)
            elif spec == "curated":
                neg = train_curated[:n_neg] if train_curated else []
            else:
                neg = _topology_stacked_negatives(train_pos, n_neg, seed, spec)
            if not neg:
                continue
            Xtr = _hadamard(emb, list(train_pos) + list(neg))
            ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(neg))]
            clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            metrics[strat]["auroc"].append(roc_auc_score(yte, p))
            metrics[strat]["auprc"].append(average_precision_score(yte, p))
            metrics[strat]["mrr"].append(_mrr(yte, p))
            for k in args.k:
                kk = min(k, len(test_pos), len(test_neg))
                metrics[strat][f"ppi@top{k}"].append(_hits_at_k(yte, p, kk, positive=True))
                metrics[strat][f"ppni@bot{k}"].append(_hits_at_k(yte, p, kk, positive=False))
            print(f"  seed {seed}  {strat:<26} AUROC={metrics[strat]['auroc'][-1]:.3f}  "
                  f"n_neg={len(neg)}")

    # Table-1-style output (full battery)
    cols = ["auroc", "auprc", "mrr"] + [f"ppi@top{k}" for k in args.k] + [f"ppni@bot{k}" for k in args.k]
    mean = lambda s, c: (float(np.mean(metrics[s][c])) if metrics[s][c] else float("nan"))
    print("\n" + "=" * 96)
    print("  Table 1 (paper style) — mean over seeds. Ranking metrics reveal tail behavior AUROC hides.")
    print("  baselines: random, curated(experimental negs) | negaverse: topology, stacked, +each rule")
    print("-" * 96)
    hdr = f"  {'strategy':<26}" + "".join(f"{c:>11}" for c in cols)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in strategies:
        row = "".join(f"{mean(s,c):>11.3f}" if metrics[s][c] else f"{'--':>11}" for c in cols)
        print(f"  {s:<26}{row}")
    print("-" * 96)
    # per-strategy deltas vs random on the metrics the paper emphasizes
    print("  Δ vs random (positive-ranking PPIHits@Top100 · negative-ranking PPNIHits@Bottom100):")
    r_top = mean("random", "ppi@top100"); r_bot = mean("random", "ppni@bot100")
    for s in strategies:
        if s == "random" or not metrics[s]["auroc"]:
            continue
        print(f"    {s:<26} PPIHits@Top100 {mean(s,'ppi@top100')-r_top:+.3f}   "
              f"PPNIHits@Bottom100 {mean(s,'ppni@bot100')-r_bot:+.3f}")


if __name__ == "__main__":
    main()
