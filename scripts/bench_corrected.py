"""Corrected negative-sampling benchmark — removes the artifacts that made the
headline "filters worse than random" result (see docs/FILTER-EFFECTIVENESS.md).

Two independent analyses converged on the same diagnosis: the −0.097 HuRI deficit
was mostly a BENCHMARK ARTIFACT, not evidence the filters pick bad negatives:

  1. AGGRESSIVE POSITIVE CAP (6,000 of 52,068 edges) → an artificially sparse
     training graph → most proteins isolated → zero SVD embeddings → the test set
     is dominated by an "either endpoint isolated ⇒ negative" shortcut. Random
     negatives (81% all-zero features) reproduce that shortcut; topology-hard
     negatives (0% isolated, since topology can't call an isolated pair hard)
     never learn it. ~85% of the deficit rode on this. Raising to 20k positives
     already flipped Δ to +0.007.
  2. UNEQUAL POOLS: random didn't get the external veto, so it accidentally
     included ~35-44 known positives and ~4-7 full-HuRI edges — dirtier than the
     veto-cleaned topology set, yet scored higher. AUROC rewarded the shortcut,
     not purity.
  3. 100% HARD-TAIL REPLACEMENT: selecting only the topology-hardest negatives is
     a narrow, hidden-positive-enriched distribution — risky even once the
     artifacts are fixed.

This bench fixes all three:
  * ONE frozen, veto-cleaned candidate pool shared by every arm (equal purity);
  * configurable, un-capped-by-default positive set (`--max-positives 0` = full);
  * five arms drawn from that same pool: raw random, veto random, topology-HARD,
    topology-SAFE (highest-confidence across the FULL pool — the representative
    clean selection the pipeline never offered), and stacked (hard tail re-ranked
    by fused biology confidence);
  * degree/coverage-stratified reporting (all pairs vs both-endpoints-non-isolated),
    which isolates the shortcut;
  * AUROC + AUPRC + PPIHits@TopK/PPNIHits@BottomK.

    PYTHONPATH=. python3 scripts/bench_corrected.py [--max-positives 20000] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression  # noqa: F401 (available for extra rows)
from negaverse.bench.benchmark import _spectral_embeddings, _hadamard_features


def _make_model(kind: str, seed: int):
    """Fresh classifier. Model-sensitivity: does the negative-selection verdict
    survive changing the downstream learner (RF -> boosted trees)?"""
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(kind)
from negaverse.candidates import generate_candidates
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space
from negaverse.rule_engine import load_rules
from negaverse.streams import build_filters
from negaverse.streams.rules import RuleGradedFilter

ARMS = ["random_raw", "random_veto", "topology_hard", "topology_safe", "stacked"]


def _score_pool(train_pos, max_pool, seed, full_pos_set, exclude=None, inject=None):
    """One frozen candidate pool, each pair scored ONCE, on the TRAIN graph only.
    `exclude` (held-out test-positive pairs, as frozensets) is removed from the
    candidate pool so a test positive can never be selected as a training negative.
    `inject` (real hidden-positive pairs) are scored and added with injected=True and
    veto BYPASSED — the injection backtest: simulated hidden positives the veto can't
    see, so we can measure which selection arm wrongly picks them as negatives."""
    exclude = exclude or set()
    inject = inject or []
    nodes = {p for e in train_pos for p in e}
    tg = TypedInteractionGraph.from_edges(
        list(train_pos), {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="pool")
    veto = build_filters("ppi", ["known_positive_veto"])[0]
    structured, topology = build_filters("ppi", ["structured", "topology"])
    # score each rule SEPARATELY so any subset's fused confidence can be recomputed
    # without re-scoring the pool (RuleGradedFilter fuses fired rules as Σ(w·v)/Σ(w)).
    per_rule = {r.id: RuleGradedFilter(rules=[r]) for r in _RULES}
    for f in [veto, structured, topology] + list(per_rule.values()):
        f.fit(tg)

    deg = dict(tg.g.degree())                                   # train-graph degree, for counterfactual matching

    def _score(u, v, injected):
        vetoed = False if injected else bool(veto.score(tg, u, v).veto)
        rule_vals = {}
        for r in _RULES:
            rs = per_rule[r.id].score(tg, u, v)
            if rs.value is not None:
                rule_vals[r.id] = (r.weight, rs.value)
        t = topology.score(tg, u, v)
        return {"u": u, "v": v, "struct": structured.score(tg, u, v).value,
                "topo": t.value, "risk": float((t.evidence or {}).get("risk", 0.0)),
                "rules": rule_vals, "vetoed": vetoed, "injected": injected,
                "degsum": deg.get(u, 0) + deg.get(v, 0),
                "dirty": vetoed or (frozenset((u, v)) in full_pos_set)}

    cand = [(u, v) for (u, v) in generate_candidates(tg, max_pool=max_pool, seed=seed)
            if frozenset((u, v)) not in exclude]
    rows = [_score(u, v, False) for (u, v) in cand]
    rows += [_score(u, v, True) for (u, v) in inject]
    risks = [r["risk"] for r in rows]
    # hardness = percentile of topology risk across the pool (matches the pipeline)
    order = np.argsort(np.argsort(risks))
    for i, r in enumerate(rows):
        r["hardness"] = order[i] / max(len(rows) - 1, 1)
        r["conf"] = _conf(r, [x.id for x in _RULES])           # default conf = all rules
    return rows


def _conf(row, rule_ids):
    """Fused mean confidence for a given rule subset (matches _fuse_confidence's
    default 'mean'): mean of {structured, topology, combined-rules-value}, skipping
    abstentions. Combined-rules-value = Σ(w·v)/Σ(w) over the subset's FIRED rules."""
    vals = [x for x in (row["struct"], row["topo"]) if x is not None]
    fired = [(w, v) for rid, (w, v) in row["rules"].items() if rid in rule_ids]
    if fired:
        num = sum(w * v for w, v in fired); den = sum(w for w, _ in fired) or len(fired)
        vals.append(num / den)
    return float(np.mean(vals)) if vals else 0.5


def _select(pool, arm, n, rng):
    clean = [r for r in pool if not r["vetoed"]]               # veto-cleaned pool
    if arm == "random_raw":
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [(pool[i]["u"], pool[i]["v"]) for i in idx]
    if arm == "random_veto":
        idx = rng.choice(len(clean), size=min(n, len(clean)), replace=False)
        return [(clean[i]["u"], clean[i]["v"]) for i in idx]
    if arm == "topology_hard":
        s = sorted(clean, key=lambda r: r["hardness"], reverse=True)
        return [(r["u"], r["v"]) for r in s[:n]]
    if arm == "topology_safe":
        s = sorted(clean, key=lambda r: r["conf"], reverse=True)   # most-confident SAFE negatives
        return [(r["u"], r["v"]) for r in s[:n]]
    if arm == "stacked":
        hard = sorted(clean, key=lambda r: r["hardness"], reverse=True)[:max(4 * n, n)]
        hard.sort(key=lambda r: r["conf"], reverse=True)          # re-rank hard tail by biology conf
        return [(r["u"], r["v"]) for r in hard[:n]]
    raise ValueError(arm)


def _select_stacked_subset(pool, n, rule_ids):
    """Stacked selection where the hard-tail re-ranking uses only `rule_ids` — for
    per-rule leave-one-out ablation."""
    clean = [r for r in pool if not r["vetoed"]]
    hard = sorted(clean, key=lambda r: r["hardness"], reverse=True)[:max(4 * n, n)]
    hard.sort(key=lambda r: _conf(r, rule_ids), reverse=True)
    return [(r["u"], r["v"]) for r in hard[:n]]


def _select_mixture(pool, n, props, rng):
    """Curriculum mixture (Park & Marcotte: training-sampling ≠ evaluation-sampling).
    props = (representative, safe, verified_hard) fractions summing to 1:
      * representative — veto-clean random (matches the eval population);
      * safe — highest fused-score negatives (cleanest);
      * verified_hard — the biology-re-ranked hard tail (our judge-free proxy for
        'verified hard'; the LLM-gated version is Tier-2 verification below).
    Buckets are de-duplicated; any shortfall is back-filled from representative."""
    clean = [r for r in pool if not r["vetoed"]]
    rep_f, safe_f, hard_f = props
    n_hard = int(round(n * hard_f)); n_safe = int(round(n * safe_f))
    n_rep = max(0, n - n_hard - n_safe)
    chosen: dict = {}
    key = lambda r: frozenset((r["u"], r["v"]))
    # verified-hard bucket: hard tail re-ranked by fused biology score (stacked-style)
    hard = sorted(clean, key=lambda r: r["hardness"], reverse=True)[:max(4 * n_hard, n_hard)]
    hard.sort(key=lambda r: r["conf"], reverse=True)
    for r in hard[:n_hard]:
        chosen[key(r)] = (r["u"], r["v"])
    # safe bucket: highest fused score not already taken
    for r in sorted(clean, key=lambda r: r["conf"], reverse=True):
        if len(chosen) >= n_hard + n_safe:
            break
        chosen.setdefault(key(r), (r["u"], r["v"]))
    # representative bucket: uniform random from what's left, then back-fill
    remaining = [r for r in clean if key(r) not in chosen]
    if remaining:
        idx = rng.choice(len(remaining), size=min(n_rep + (n - len(chosen)), len(remaining)),
                         replace=False)
        for i in idx:
            if len(chosen) >= n:
                break
            chosen[key(remaining[i])] = (remaining[i]["u"], remaining[i]["v"])
    return list(chosen.values())[:n]


def _degree_match(clean, pos_degsums, n, rng):
    """Greedy 1:1 nearest-degree-sum match of `clean` candidates to the positives'
    degree distribution (the propensity/nuisance covariate we match on)."""
    buckets: dict = {}
    for r in clean:
        buckets.setdefault(r["degsum"], []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys = np.array(sorted(buckets))
    if len(keys) == 0:
        return []
    out = []
    targets = list(pos_degsums)
    rng.shuffle(targets)
    for d in targets:
        if len(out) >= n:
            break
        k = keys[np.argmin(np.abs(keys - d))]                   # nearest available degsum bucket
        while buckets.get(k) is not None and not buckets[k]:
            keys = keys[keys != k]
            if len(keys) == 0:
                return [(r["u"], r["v"]) for r in out][:n]
            k = keys[np.argmin(np.abs(keys - d))]
        out.append(buckets[k].pop())
    return [(r["u"], r["v"]) for r in out][:n]


def _select_counterfactual(pool, pos_degsums, n, rng):
    """Matched-counterfactual (Tier 3): degree-match to positives from the WHOLE
    veto-clean pool. Loses — matching pulls in positive-like pairs that are hidden
    positives (measured ~48% contamination). The problem this PSM fixes."""
    return _degree_match([r for r in pool if not r["vetoed"]], pos_degsums, n, rng)


def _select_psm(pool, pos_degsums, n, rng, cap):
    """Propensity-score matching (the fix). Same degree-matching to positives, but the
    candidate pool is restricted to the VERIFIED-CLEAN region first: veto-clean AND
    hardness ≤ cap (the injection backtest measured that hidden positives concentrate
    in the high-hardness/topology-risk tail, so capping hardness ≈ 0 contamination).
    Safety authorises the label (clean pool); matching only adds difficulty. Lowering
    `cap` trades hardness for purity — the sweep shows the trade-off."""
    clean = [r for r in pool if not r["vetoed"] and r["hardness"] <= cap]
    return _degree_match(clean, pos_degsums, n, rng)


def _hits(y, s, k, positive=True):
    order = np.argsort(s)
    idx = order[-k:] if positive else order[:k]
    return float(np.mean(y[idx] == (1 if positive else 0)))


def _evaluate(tr_pos, test_pos, train_neg, nodes, gold_test, seed, models, emb_dim=32):
    """Returns {model_kind: metric_dict}. The train/test positive split is fixed by
    the caller (fit ONCE, before pool scoring) — this only trains + scores. Same
    features/split across models; only the learner changes (model-sensitivity).

    gold_test=[] (no independent gold-negative benchmark for this dataset, e.g.
    sars/covid — see _load_dataset) -> NaN metrics rather than a fabricated
    comparison; the injection test's primary selection-rate metric doesn't call
    this at all, so it's unaffected."""
    n_bal = min(len(test_pos), len(gold_test))
    if n_bal == 0:
        nan = float("nan")
        return {m: {"auroc": nan, "auprc": nan, "ppi@100": nan, "ppni@100": nan}
                for m in models}
    test_pos, test_neg = list(test_pos)[:n_bal], list(gold_test)[:n_bal]

    train_G = nx.Graph(); train_G.add_nodes_from(nodes); train_G.add_edges_from(tr_pos)
    emb, idx, _ = _spectral_embeddings(train_G, nodes, emb_dim, seed)
    feat = lambda pairs: _hadamard_features(emb, idx, pairs)
    deg = dict(train_G.degree())
    Xtr = feat(list(tr_pos) + list(train_neg))
    ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(train_neg))]
    te_pairs = list(test_pos) + list(test_neg)
    Xte = feat(te_pairs)
    yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))].astype(int)
    keep = np.array([deg.get(u, 0) > 0 and deg.get(v, 0) > 0 for (u, v) in te_pairs])

    result = {}
    for kind in models:
        clf = _make_model(kind, seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        out = {"auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p),
               "ppi@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), True),
               "ppni@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), False)}
        if keep.sum() > 10 and len(set(yte[keep])) == 2:
            out["auroc_noniso"] = roc_auc_score(yte[keep], p[keep])
        result[kind] = out
    return result


def _load_dataset(name):
    """(all_pos edges, gold-negative pairs, node list, full_pos_set). `gold`
    is the independent negative benchmark _evaluate() scores test positives
    against — NOT the negatives any selection arm is allowed to pick from
    (those come from the frozen candidate pool instead)."""
    if name == "huri":
        g = load_huri_graph()
        gold = [tuple(p) for p in load_negatome_in_ensembl_space(set(g.g.nodes()))]
        pos = [tuple(e) for e in g.g.edges()]
        return pos, gold, list(g.g.nodes()), {frozenset(e) for e in g.g.edges()}
    if name == "dryad":
        pos, neg = [], []
        with open("local-docs/dryad-ppi/benchmarks/positives_and_negatives.tsv") as fh:
            next(fh)
            for line in fh:
                pair, cat = line.rstrip("\n").split("\t")
                a, b = pair.split("_")
                (pos if cat == "positive" else neg).append((a, b))
        nodes = sorted({p for e in pos + neg for p in e})
        return pos, neg, nodes, {frozenset(e) for e in pos}
    if name == "upna":
        # UPNA-PPI ships gene-symbol-keyed CSVs; remap to UniProt (scripts/
        # build_gene_symbol_uniprot_map.py) so this reuses the same UniProt-space
        # annotation/known-positive sources as dryad/sars instead of needing its
        # own gene-symbol-keyed copies of all of them. Scoped to the ~5,037-protein
        # TPPNI universe (see negaverse/viz/__main__.py::_load_upna_graph for why —
        # same reasoning applies here). TPPNI (their own headline contrastive-L3
        # hard-negative method) doubles as `gold`: an independent, external
        # negative benchmark this repo didn't select, exactly what `gold` needs to be.
        import pandas as pd
        upna_dir = Path("local-docs/upna-ppi")
        map_path = Path("local-docs/mappings/gene_symbol_to_uniprot.tsv")
        sym2acc: dict[str, str] = {}
        if map_path.exists():
            for line in map_path.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "\t" in line:
                    sym, acc = line.split("\t")[:2]
                    sym2acc[sym] = acc

        def _pairs(pattern: str) -> list[tuple[str, str]]:
            out = []
            for f in sorted(upna_dir.glob(pattern)):
                for chunk in pd.read_csv(f, usecols=["SymbolA", "SymbolB"], chunksize=300_000):
                    for a, b in zip(chunk["SymbolA"].astype(str), chunk["SymbolB"].astype(str)):
                        if a != b:
                            out.append((a, b))
            return out

        universe_sym = {s for pair in _pairs("TPPNI_*.csv") for s in pair}

        def _map(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
            return [(sym2acc[a], sym2acc[b]) for a, b in pairs
                    if a in universe_sym and b in universe_sym and a in sym2acc and b in sym2acc]

        pos = _map(_pairs("PPI_part_*.csv"))
        gold = _map(_pairs("TPPNI_*.csv"))
        nodes = sorted({sym2acc[s] for s in universe_sym if s in sym2acc})
        return pos, gold, nodes, {frozenset(e) for e in pos}
    if name in ("sars", "covid"):
        # Host-viral: no independent gold-negative benchmark exists for this
        # cross-species space (Negatome is human-human only — see
        # negaverse/io/negatome.py's own docstring: "NOT the viral-host
        # SARS-CoV-2 graph"). gold=[] — _evaluate() reports NaN for the
        # downstream-AUROC "damage" columns rather than fabricate a comparison,
        # but the injection test's primary metric (selection rate) needs no
        # gold at all, so it still runs. Node typing collapses to a plain
        # "protein" pool either way (_score_pool always re-types), so this
        # doesn't hit the viral/host applies_to mismatch the live pipeline has.
        from negaverse.io import load_sars_cov2_graph
        g = load_sars_cov2_graph()
        pos = [tuple(e) for e in g.g.edges()]
        return pos, [], list(g.g.nodes()), {frozenset(e) for e in pos}
    raise ValueError(name)


def _injection_backtest(all_pos, gold, nodes, full_pos_set, args):
    """Hidden-positive injection backtest (Tier 3). Inject K real interactions that
    are absent from the training graph (veto-bypassed = truly hidden), then measure
    what fraction each selection arm wrongly picks as a training negative. The claim
    negaverse makes is that it avoids hidden positives; this tests it directly.
    Lower `selected%` = better. Expectation: topology_hard picks the MOST (hidden
    positives are positive-like ⇒ hard); safe/stacked/counterfactual pick fewer."""
    arms = ["random_veto", "topology_hard", "topology_safe", "stacked", "counterfactual",
            "psm[cap=0.7]", "psm[cap=0.5]"]
    rates = {a: [] for a in arms}
    dmg = {(a, m): [] for a in arms for m in args.models}       # downstream AUROC on the injected pool
    print(f"INJECTION BACKTEST — inject K={args.inject_k} hidden positives (veto-bypassed).\n"
          f"  selection rate = fraction each arm picks as a negative (model-INDEPENDENT — selection\n"
          f"  happens before any model); AUROC = downstream damage under each learner (models={args.models}).\n"
          + "=" * 92)
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        n_test = int(len(pos) * 0.2)
        test_pos, tr_pos = pos[:n_test], pos[n_test:]
        tr_set = {frozenset(e) for e in tr_pos}
        tr_nodes = {n for e in tr_pos for n in e}               # only nodes the train graph knows
        # hidden positives: real edges NOT in the training graph, both endpoints in the train graph
        held = [tuple(p) for p in full_pos_set
                if p not in tr_set and set(p) <= tr_nodes and len(p) == 2]
        rng.shuffle(held)
        inject = held[:args.inject_k]
        inj_keys = {frozenset(p) for p in inject}
        gold_s = [p for p in gold if frozenset(p) not in full_pos_set]
        pool = _score_pool(tr_pos, args.max_pool, seed, full_pos_set,
                           exclude={frozenset(p) for p in test_pos}, inject=inject)
        n_inj_pool = sum(1 for r in pool if r.get("injected"))
        _dg = dict(nx.Graph(tr_pos).degree())
        pos_degsums = [_dg.get(u, 0) + _dg.get(v, 0) for (u, v) in tr_pos]
        n_neg = len(tr_pos)
        for a in arms:
            if a == "counterfactual":
                neg = _select_counterfactual(pool, pos_degsums, n_neg, np.random.default_rng(seed))
            elif a.startswith("psm[cap="):
                cap = float(a.split("=")[1].rstrip("]"))
                neg = _select_psm(pool, pos_degsums, n_neg, np.random.default_rng(seed), cap)
            else:
                neg = _select(pool, a, n_neg, np.random.default_rng(seed))
            sel_inj = sum(1 for uv in neg if frozenset(uv) in inj_keys)
            rates[a].append(sel_inj / max(n_inj_pool, 1))
            res = _evaluate(tr_pos, test_pos, neg, nodes, gold_s, seed, args.models)  # damage per model
            for m in args.models:
                dmg[(a, m)].append(res[m]["auroc"])
            au = "  ".join(f"{m}={res[m]['auroc']:.3f}" for m in args.models)
            print(f"  seed {seed}  {a:<16} selected {sel_inj}/{n_inj_pool} injected  AUROC[{au}]")
    print("-" * 92)
    mm = lambda a, m: float(np.mean(dmg[(a, m)])) if dmg[(a, m)] else float("nan")
    hdr = f"  {'arm':<16}{'hidden-pos selected%':>22}" + "".join(f"{'AUROC('+m+')':>14}" for m in args.models)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for a in arms:
        print(f"  {a:<16}{100*float(np.mean(rates[a])):>21.1f}%"
              + "".join(f"{mm(a, m):>14.3f}" for m in args.models))
    print("-" * 92)
    print("  selection% is identical across models (selection precedes the learner). The AUROC")
    print("  columns show the damage the contamination does downstream — model-robust if both agree.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad", "upna", "sars", "covid"], default="huri")
    ap.add_argument("--max-positives", type=int, default=20000, help="0 = all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=200000)
    ap.add_argument("--models", nargs="+", default=["rf", "lgbm"], choices=["rf", "lgbm"])
    ap.add_argument("--rule-ablation", action="store_true",
                    help="leave each graded rule out of the stacked arm one-by-one")
    ap.add_argument("--mixture", action="store_true",
                    help="sweep representative/safe/verified-hard mixture proportions")
    ap.add_argument("--injection-test", action="store_true",
                    help="inject K known hidden positives; report which arm wrongly selects them")
    ap.add_argument("--inject-k", type=int, default=1000)
    ap.add_argument("--psm", action="store_true",
                    help="propensity-score-matched negatives (clean-pool degree match); sweep hardness cap")
    ap.add_argument("--eval-match", action="store_true",
                    help="degree-match negatives to a HELD-OUT fold of the gold negatives (eval population)")
    args = ap.parse_args()

    global _RULES
    _RULES = [r for r in load_rules()
              if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]

    # arm_list + spec: "builtin" -> _select; None -> veto random; list[str] -> stacked with that rule subset
    if args.rule_ablation:
        all_ids = [r.id for r in _RULES]
        spec = {"random_veto": None, "stacked[ALL rules]": all_ids}
        for r in _RULES:
            spec[f"stacked[-{r.id}]"] = [x for x in all_ids if x != r.id]
        spec["stacked[NO rules]"] = []
    elif args.eval_match:
        spec = {"random_veto": "builtin", "topology_safe": "builtin", "stacked": "builtin",
                "psm_to_positives": ("psm", 0.7), "eval_matched": ("evalmatch",),
                "eval_matched_clean": ("evalmatch_clean", 0.7)}
    elif args.psm:
        spec = {"random_veto": "builtin", "topology_safe": "builtin", "stacked": "builtin",
                "counterfactual (cap=1.0)": ("cf",)}
        for cap in [0.9, 0.7, 0.5]:
            spec[f"psm[cap={cap}]"] = ("psm", cap)
    elif args.mixture:
        spec = {"random_veto": "builtin", "topology_safe": "builtin", "stacked": "builtin",
                "counterfactual": ("cf",)}
        for rep, safe, hard in [(0.60, 0.30, 0.10), (0.70, 0.20, 0.10), (0.50, 0.30, 0.20)]:
            spec[f"mix[{int(rep*100)}/{int(safe*100)}/{int(hard*100)}]"] = ("mix", (rep, safe, hard))
    else:
        spec = {a: "builtin" for a in ARMS}
    arm_list = list(spec)

    all_pos, gold, nodes, full_pos_set = _load_dataset(args.dataset)
    print("=" * 92)
    print(f"CORRECTED benchmark — {args.dataset.upper()}, one frozen veto-cleaned pool"
          f"{'  [RULE ABLATION]' if args.rule_ablation else ''}")
    print(f"full edges={len(all_pos)}  nodes={len(nodes)}  max_positives={args.max_positives or 'ALL'}  "
          f"gold={len(gold)}  seeds={args.seeds}  rules={[r.id for r in _RULES]}")
    # fail-closed certification: were the negatives actually screened against external DBs?
    _v = build_filters("ppi", ["known_positive_veto"])[0]
    _v.fit(TypedInteractionGraph.from_edges(
        all_pos, {n: "protein" for n in nodes}, admissible_types=[("protein", "protein")], name="cert"))
    _c = _v.certification()
    print(f"veto certification: {'CERTIFIED' if _c['certified'] else '*** UNCERTIFIED ***'}  "
          f"loaded={_c['loaded']}  missing={_c['missing']}")
    if not _c["certified"]:
        print("  WARNING: no external known-positive DB screened this pool — hidden-positive "
              "leakage from BioGRID/IntAct is NOT ruled out. Purity numbers below are graph-only.")
    print("=" * 92)

    if args.injection_test:
        _injection_backtest(all_pos, gold, nodes, full_pos_set, args)
        return

    # metrics[(arm, model)][metric] = [per-seed values]
    metrics = {(a, m): {k: [] for k in ["auroc", "auprc", "ppi@100", "ppni@100", "auroc_noniso"]}
               for a in arm_list for m in args.models}
    purity = {a: [] for a in arm_list}
    print(f"models: {args.models}")
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        # SPLIT ONCE, before scoring — the selector fits on train edges only (no leakage)
        pos = [pos[i] for i in rng.permutation(len(pos))]
        n_test = int(len(pos) * 0.2)
        test_pos, tr_pos = pos[:n_test], pos[n_test:]
        test_excl = {frozenset(p) for p in test_pos}
        gold_s = [p for p in gold if frozenset(p) not in full_pos_set]
        rng.shuffle(gold_s)
        # eval-match: split gold into a MATCH fold (target distribution) and a disjoint TEST fold
        gold_eval = gold_s
        match_degsums = []
        if args.eval_match:
            h = len(gold_s) // 2
            match_gold, gold_eval = gold_s[:h], gold_s[h:]
            _dgm = dict(nx.Graph(tr_pos).degree())
            match_degsums = [_dgm.get(u, 0) + _dgm.get(v, 0) for (u, v) in match_gold]
        pool = _score_pool(tr_pos, args.max_pool, seed, full_pos_set, exclude=test_excl)
        n_neg = int(len(tr_pos))                                  # one negative per training positive
        _dg = dict(nx.Graph(tr_pos).degree())
        pos_degsums = [_dg.get(u, 0) + _dg.get(v, 0) for (u, v) in tr_pos]
        for arm in arm_list:
            sp = spec[arm]
            if sp == "builtin":
                neg = _select(pool, arm, n_neg, np.random.default_rng(seed))
            elif sp is None:
                neg = _select(pool, "random_veto", n_neg, np.random.default_rng(seed))
            elif isinstance(sp, tuple) and sp[0] == "mix":
                neg = _select_mixture(pool, n_neg, sp[1], np.random.default_rng(seed))
            elif isinstance(sp, tuple) and sp[0] == "cf":
                neg = _select_counterfactual(pool, pos_degsums, n_neg, np.random.default_rng(seed))
            elif isinstance(sp, tuple) and sp[0] == "psm":
                neg = _select_psm(pool, pos_degsums, n_neg, np.random.default_rng(seed), sp[1])
            elif isinstance(sp, tuple) and sp[0] in ("evalmatch", "evalmatch_clean"):
                # resample the held-out gold-negative degree distribution up to n_neg targets
                _r = np.random.default_rng(seed)
                targets = list(_r.choice(match_degsums, size=n_neg)) if match_degsums else []
                clean = ([r for r in pool if not r["vetoed"]] if sp[0] == "evalmatch"
                         else [r for r in pool if not r["vetoed"] and r["hardness"] <= sp[1]])
                neg = _degree_match(clean, targets, n_neg, _r)
            else:
                neg = _select_stacked_subset(pool, n_neg, sp)
            if not neg:
                continue
            dirty = sum(1 for u, v in neg if frozenset((u, v)) in full_pos_set)
            purity[arm].append(dirty)
            per_model = _evaluate(tr_pos, test_pos, neg, nodes, gold_eval, seed, args.models)
            for m, res in per_model.items():
                for k, val in res.items():
                    if k in metrics[(arm, m)]:
                        metrics[(arm, m)][k].append(val)
            au = "  ".join(f"{m}={per_model[m]['auroc']:.3f}" for m in args.models)
            print(f"  seed {seed}  {arm:<28} AUROC[{au}]  n_neg={len(neg)} hidden_pos={dirty}")

    mean = lambda a, m, k: float(np.mean(metrics[(a, m)][k])) if metrics[(a, m)][k] else float("nan")
    w = 34 if args.rule_ablation else 15
    ref = "stacked[ALL rules]" if args.rule_ablation else "random_veto"
    for m in args.models:
        print("\n" + "=" * 100)
        print(f"  CORRECTED results — model={m.upper()}, mean over seeds. 'noniso' = both-endpoints-non-isolated.")
        print("-" * 100)
        hdr = f"  {'arm':<{w}}{'AUROC':>9}{'AUPRC':>9}{'AUROC_noniso':>14}{'PPIHit@100':>12}{'PPNIHit@100':>13}{'hidden+':>9}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for a in arm_list:
            hp = float(np.mean(purity[a])) if purity[a] else float("nan")
            print(f"  {a:<{w}}{mean(a,m,'auroc'):>9.3f}{mean(a,m,'auprc'):>9.3f}"
                  f"{mean(a,m,'auroc_noniso'):>14.3f}{mean(a,m,'ppi@100'):>12.3f}"
                  f"{mean(a,m,'ppni@100'):>13.3f}{hp:>9.1f}")
        print("-" * 100)
        if args.rule_ablation:
            ra, rn = mean(ref, m, "auroc"), mean(ref, m, "auroc_noniso")
            print(f"  Δ vs {ref} (a rule that HELPS the stack shows a NEGATIVE Δ when removed):")
            for a in arm_list:
                if a.startswith("stacked[-"):
                    print(f"    {a:<34} Δ AUROC {mean(a,m,'auroc')-ra:+.4f}   "
                          f"Δ noniso {mean(a,m,'auroc_noniso')-rn:+.4f}")
        else:
            r, rn = mean("random_veto", m, "auroc"), mean("random_veto", m, "auroc_noniso")
            others = [a for a in arm_list if a != "random_veto"]
            print("  Δ AUROC vs veto-random:  " + "   ".join(
                f"{a} {mean(a,m,'auroc')-r:+.3f}" for a in others))
            print("  Δ AUROC_noniso vs veto-random:  " + "   ".join(
                f"{a} {mean(a,m,'auroc_noniso')-rn:+.3f}" for a in others))
    print("\n  hidden+ = mean # true positives leaked into the negative set (purity; lower=cleaner)")


if __name__ == "__main__":
    main()
