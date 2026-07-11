"""Ablation bench for biology signals — how much does each one contribute?

Scores every signal on real interactions vs two kinds of negative, side by side,
because the contrast is the whole point:

  * vs RANDOM non-edges  — the easy case (random pairs differ in everything);
  * vs HARD negatives    — biologically/structurally validated non-interactors,
                           which are *hard* (often co-localized, co-annotated
                           pairs that still don't bind). Negatome for HuRI;
                           DRYAD's own controls for the structure dataset.

A signal strong vs random but ~0.5 vs hard only finds *easy* negatives; a signal
that helps vs hard cracks the interesting ones. Per signal we report coverage,
AUROC vs random, AUROC vs hard, and (for rules) leave-one-out Δ.

Signal families in the table:
  * YAML biology rules (rules/*.yaml) — auto-loaded;
  * derived "tool" signals (graded overlaps) — smarter use of the same data;
  * co-evolution / structure — an ESM2 signal (an evolutionary-scale model, so
    cosine captures evolutionary + structural relatedness) on the DRYAD dataset;
  * external — any precomputed `id<TAB>id<TAB>score` file (real AF2-Multimer
    interface score, EVcouplings, STRING, …) via --external name=path.tsv.

Separation vs independent ground truth — an honest proxy; the feature-independent
downstream benchmark (negaverse.bench) is the final word.

    PYTHONPATH=. python3 scripts/bench_rules.py                    # HuRI + Negatome
    PYTHONPATH=. python3 scripts/bench_rules.py --dataset dryad    # + ESM2 co-evolution column
    PYTHONPATH=. python3 scripts/bench_rules.py --external af2=out/af2_scores.tsv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space
from negaverse.io.annotations import build_annotation_table
from negaverse.rule_engine import load_rules
from negaverse.streams.rules import RuleGradedFilter, _RuleFilterBase

_DRYAD = "local-docs/dryad-ppi"


def _random_nonedges(nodes, pos_set, n, rng):
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < n * 80 + 1000:
        tries += 1
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a != b and k not in pos_set and k not in seen:
            seen.add(k); out.append((a, b))
    return out


def _auroc(y, s):
    s = np.where(np.isnan(s), 0.5, s)
    return round(float(roc_auc_score(y, s)), 3) if len(set(y)) == 2 else None


# ---- datasets ----------------------------------------------------------
def _load_huri(seed, n_each):
    graph = load_huri_graph()
    rng = np.random.default_rng(seed)
    nodes = list(graph.g.nodes()); node_set = set(nodes)
    pos_all = [tuple(e) for e in graph.g.edges()]
    pos_set = {frozenset(e) for e in pos_all}
    hard = [tuple(p) for p in load_negatome_in_ensembl_space(node_set)
            if set(p) <= node_set and frozenset(p) not in pos_set]
    rng.shuffle(pos_all); rng.shuffle(hard)
    n = min(len(hard), n_each, len(pos_all))
    pos = pos_all[:n]
    return graph, pos, hard[:n], _random_nonedges(nodes, pos_set, n, rng), None


def _load_huintaf2(seed, n_each):
    """Hu.MAP co-complex interactions (positives) vs random pairs (negatives),
    both UniProt-keyed with real AF2 pDockQ (via --external af2). The single
    negative set is genuinely random, so 'vs random' and 'vs hard' coincide here."""
    import csv as _csv
    rng = np.random.default_rng(seed)

    def pairs(name):
        out = []
        with open(f"local-docs/huintaf2/{name}.csv") as fh:
            for r in _csv.DictReader(fh):
                p = r["Name"].split("-")
                if len(p) == 2:
                    out.append((p[0], p[1]))
        return out

    pos, neg = pairs("humap"), pairs("random")
    rng.shuffle(pos); rng.shuffle(neg)
    n = min(len(neg), n_each, len(pos))
    pos, neg = pos[:n], neg[:n]
    nodes = sorted({p for pr in pos + neg for p in pr})
    graph = TypedInteractionGraph.from_edges(
        pos, {p: "protein" for p in nodes},
        admissible_types=[("protein", "protein")], name="huintaf2")
    return graph, pos, neg, neg, None          # random == hard: only random negatives exist


def _load_dryad(seed, n_each):
    from negaverse.io import load_embeddings_npz
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    with open(f"{_DRYAD}/benchmarks/benchmarks/positives_and_negatives.tsv") as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    emb = load_embeddings_npz(f"{_DRYAD}/esm2_t6_emb.npz")
    nodes = sorted({p for pr in pos + neg for p in pr})
    node_type = {n: "protein" for n in nodes}
    graph = TypedInteractionGraph.from_edges(
        pos, node_type, admissible_types=[("protein", "protein")], name="dryad")
    pos_set = {frozenset(e) for e in pos} | {frozenset(e) for e in neg}
    rng.shuffle(pos); rng.shuffle(neg)
    n = min(len(neg), n_each, len(pos))
    return graph, pos[:n], neg[:n], _random_nonedges(nodes, pos_set, n, rng), emb


# ---- signal scorers ----------------------------------------------------
def _rule_scorer(rules, ann, graph):
    f = RuleGradedFilter(rules=rules if isinstance(rules, list) else [rules],
                         annotations=ann)
    f.fit(graph)
    def score(pairs):
        out = np.full(len(pairs), np.nan)
        for i, (u, v) in enumerate(pairs):
            val = f.score(graph, u, v).value
            if val is not None:
                out[i] = val
        return out
    return score


def _tool_scorers(ann):
    def jac(a, b):
        a, b = set(a or ()), set(b or ())
        return len(a & b) / len(a | b) if (a or b) else None

    def make(field, transform):
        def score(pairs):
            out = np.full(len(pairs), np.nan)
            for i, (u, v) in enumerate(pairs):
                val = transform(ann.get(u, {}).get(field), ann.get(v, {}).get(field))
                if val is not None:
                    out[i] = float(val)
            return out
        return score

    return {
        "tool:compartment_jaccard": make("compartments", lambda a, b: None if jac(a, b) is None else 1 - jac(a, b)),
        "tool:process_jaccard": make("processes", lambda a, b: None if jac(a, b) is None else 1 - jac(a, b)),
        "tool:hydrophobicity_gap": make("surface_hydrophobicity",
                                        lambda a, b: None if a is None or b is None else min(1.0, abs(a - b) / 0.5)),
    }


def _esm2_coevolution_scorer(emb):
    """Co-evolution / structure proxy: ESM2-cosine *dissimilarity* — lower cosine
    (less evolutionarily/structurally related) => higher safe-negative confidence."""
    norm = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in emb.items()}
    def score(pairs):
        out = np.full(len(pairs), np.nan)
        for i, (u, v) in enumerate(pairs):
            a, b = norm.get(u), norm.get(v)
            if a is not None and b is not None:
                out[i] = float((1.0 - a @ b) / 2.0)
        return out
    return score


def _external_scorer(path):
    table: dict[frozenset, float] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                table[frozenset((parts[0], parts[1]))] = float(parts[2])
    def score(pairs):
        out = np.full(len(pairs), np.nan)
        for i, (u, v) in enumerate(pairs):
            val = table.get(frozenset((u, v)))
            if val is not None:
                out[i] = val
        return out
    return score


def main():
    ap = argparse.ArgumentParser(prog="bench_rules")
    ap.add_argument("--dataset", choices=["huri", "dryad", "huintaf2"], default="huri")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-each", type=int, default=400)
    ap.add_argument("--external", action="append", default=[],
                    help="name=path.tsv (id<TAB>id<TAB>score) — e.g. af2=out/af2.tsv")
    args = ap.parse_args()

    load = {"dryad": _load_dryad, "huintaf2": _load_huintaf2}.get(args.dataset, _load_huri)
    graph, pos, hard, rand, emb = load(args.seed, args.n_each)
    n = len(pos)
    pairs_rand, pairs_hard = pos + rand, pos + hard
    y = np.r_[np.zeros(n), np.ones(n)]            # 1 = negative

    ann = _RuleFilterBase._augment_with_graph(build_annotation_table(), graph)
    rules = [r for r in load_rules()
             if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]

    cov = lambda s: round(float(np.mean(~np.isnan(s))), 3)
    rows: dict[str, dict] = {}

    for r in rules:
        sc = _rule_scorer(r, ann, graph)
        rows[r.id] = {"coverage": cov(sc(pairs_hard)),
                      "auroc_vs_random": _auroc(y, sc(pairs_rand)),
                      "auroc_vs_hard": _auroc(y, sc(pairs_hard))}
    for name, fn in _tool_scorers(ann).items():
        rows[name] = {"coverage": cov(fn(pairs_hard)),
                      "auroc_vs_random": _auroc(y, fn(pairs_rand)),
                      "auroc_vs_hard": _auroc(y, fn(pairs_hard))}
    if emb is not None:
        fn = _esm2_coevolution_scorer(emb)
        rows["coev:esm2_cosine"] = {"coverage": cov(fn(pairs_hard)),
                                    "auroc_vs_random": _auroc(y, fn(pairs_rand)),
                                    "auroc_vs_hard": _auroc(y, fn(pairs_hard))}
    for spec in args.external:
        name, path = spec.split("=", 1)
        fn = _external_scorer(path)
        rows[f"ext:{name}"] = {"coverage": cov(fn(pairs_hard)),
                               "auroc_vs_random": _auroc(y, fn(pairs_rand)),
                               "auroc_vs_hard": _auroc(y, fn(pairs_hard))}

    combined_rand = _auroc(y, _rule_scorer(rules, ann, graph)(pairs_rand))
    combined_hard = _auroc(y, _rule_scorer(rules, ann, graph)(pairs_hard))
    for r in rules:
        rest = [x for x in rules if x.id != r.id]
        loo = _auroc(y, _rule_scorer(rest, ann, graph)(pairs_hard)) if rest else 0.5
        rows[r.id]["delta_hard_if_removed"] = round((combined_hard or 0) - (loo or 0), 3)

    report = {"dataset": args.dataset, "n_per_class": n,
              "combined_auroc_vs_random": combined_rand,
              "combined_auroc_vs_hard": combined_hard, "signals": rows}
    os.makedirs("out", exist_ok=True)
    with open(f"out/rules_bench_{args.dataset}.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 84)
    print(f"Biology-signal ablation [{args.dataset}] — interactions vs negatives ({n}/class)")
    print("=" * 84)
    hdr = f"{'signal':<36}{'cover':>7}{'vs random':>11}{'vs hard':>9}{'Δhard':>8}"
    print(hdr); print("-" * len(hdr))
    for name, m in sorted(rows.items(), key=lambda kv: -(kv[1].get("auroc_vs_hard") or 0)):
        r = "abstain" if m["auroc_vs_random"] is None else f"{m['auroc_vs_random']:.3f}"
        g = "abstain" if m["auroc_vs_hard"] is None else f"{m['auroc_vs_hard']:.3f}"
        d = f"{m['delta_hard_if_removed']:+.3f}" if "delta_hard_if_removed" in m else ""
        print(f"{name:<36}{m['coverage']*100:>6.0f}%{r:>11}{g:>9}{d:>8}")
    print("-" * len(hdr))
    print(f"{'ALL RULES COMBINED':<36}{'':>7}{combined_rand:>11.3f}{combined_hard:>9.3f}")
    print("\n  vs random = easy negatives · vs hard = validated non-interactors ·"
          " 0.5 = no signal, <0.5 = mis-scores")
    print(f"  (full report -> out/rules_bench_{args.dataset}.json)")


if __name__ == "__main__":
    main()
