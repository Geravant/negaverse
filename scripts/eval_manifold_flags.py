"""Does the manifold filter's flag actually catch hidden false negatives?

The honest validity test for the `suspected_false_negative` flag, graded against
an *independent* yardstick — not the embedding it is built from.

Setup (leakage-free):
  * split HuRI positives into train / held-out;
  * fit the manifold (and topology, for comparison) on the TRAIN graph only —
    the held-out edges never enter the embedding;
  * the held-out edges are "hidden positives": real interactions the pipeline
    does not know about. If a negative-generator emits one, that's a false
    negative — exactly what the flag should catch.
  * random non-edges (in no positive set) are the "clean" negatives.

The ground-truth label (hidden-positive vs clean) comes from the held-out split,
NOT from the embedding, so this is not the circular "grade with the same ruler"
trap (docs/BENCHMARK-FINDINGS.md). We ask three things:
  1. Does the flag separate hidden positives from clean negatives? (AUROC, and a
     flag lift = flag-rate on hidden ÷ flag-rate on clean.)
  2. Is it better than / complementary to topology, which we already have?
  3. In a realistic eval set lightly contaminated with hidden positives, does
     dropping the flagged pairs remove more contamination than clean pairs?

    PYTHONPATH=. python3 scripts/eval_manifold_flags.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from negaverse.io import load_huri_graph
from negaverse.graph import TypedInteractionGraph
from negaverse.streams.manifold import ManifoldSurprisalFilter
from negaverse.streams.topology import TopologyFilter


def _random_nonedges(nodes, pos_set, n, rng):
    seen, out = set(), []
    N = len(nodes)
    tries, cap = 0, n * 60 + 1000
    while len(out) < n and tries < cap:
        tries += 1
        a, b = nodes[rng.integers(N)], nodes[rng.integers(N)]
        if a == b:
            continue
        key = frozenset((a, b))
        if key in pos_set or key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def main(seed: int = 0, test_frac: float = 0.2, n_each: int = 1500,
         contamination: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    graph = load_huri_graph()
    pos = [tuple(e) for e in graph.g.edges()]
    pos_set = {frozenset(e) for e in pos}
    nodes = list(graph.g.nodes())

    rng.shuffle(pos)
    n_test = int(len(pos) * test_frac)
    held_out, train_pos = pos[:n_test], pos[n_test:]

    # fit both filters on the TRAIN graph only (held-out edges never seen)
    tg = TypedInteractionGraph.from_edges(
        train_pos, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="huri-train")
    manifold = ManifoldSurprisalFilter(seed=seed); manifold.fit(tg)
    topo = TopologyFilter(); topo.fit(tg)

    hidden = held_out[:]
    clean = _random_nonedges(nodes, pos_set, len(hidden), rng)

    # score every pair; keep only those the manifold can actually place
    def collect(pairs):
        res, mval, mflag, trisk = [], [], [], []
        for u, v in pairs:
            ms = manifold.score(tg, u, v)
            if ms.value is None:                       # abstains (isolated node)
                res.append(None); continue
            res.append(ms.evidence["resemblance"])
            mval.append(ms.value)
            mflag.append("suspected_false_negative" in ms.flags)
            ts = topo.score(tg, u, v)
            trisk.append((ts.evidence or {}).get("risk", 0.0))
        return res, np.array(mflag), np.array(trisk)

    hid_res, hid_flag, hid_trisk = collect(hidden)
    cln_res, cln_flag, cln_trisk = collect(clean)
    hid_ok = [r for r in hid_res if r is not None]
    cln_ok = [r for r in cln_res if r is not None]
    abst = 1 - (len(hid_ok) + len(cln_ok)) / max(len(hid_res) + len(cln_res), 1)

    y = np.r_[np.ones(len(hid_ok)), np.zeros(len(cln_ok))]     # 1 = hidden positive
    resemblance = np.r_[hid_ok, cln_ok]
    trisk = np.r_[hid_trisk, cln_trisk]

    auroc_manifold = round(float(roc_auc_score(y, resemblance)), 4)
    auroc_topology = round(float(roc_auc_score(y, trisk)), 4)
    flag_lift = round(float(hid_flag.mean() / max(cln_flag.mean(), 1e-9)), 2)

    # complementarity: on the pairs topology calls "safe" (below-median risk),
    # can the manifold still find the hidden positives it missed?
    med = float(np.median(trisk))
    safe = trisk <= med
    auroc_on_topo_safe = (round(float(roc_auc_score(y[safe], resemblance[safe])), 4)
                          if len(set(y[safe])) == 2 else None)

    # realistic contamination scenario: an eval set that is mostly clean with a
    # small fraction of hidden positives sneaked in; drop the flagged pairs.
    n_clean_eval = len(cln_ok)
    n_contam = max(1, int(n_clean_eval * contamination))
    contam_idx = rng.choice(len(hid_flag), size=min(n_contam, len(hid_flag)), replace=False)
    dropped_contam = int(hid_flag[contam_idx].sum())
    dropped_clean = int(cln_flag.sum())
    before = n_contam
    after = n_contam - dropped_contam

    report = {
        "n_hidden_scored": len(hid_ok), "n_clean_scored": len(cln_ok),
        "abstention_rate": round(abst, 4),
        "auroc_flag_signal_manifold": auroc_manifold,
        "auroc_topology_baseline": auroc_topology,
        "flag_lift_hidden_vs_clean": flag_lift,
        "auroc_manifold_on_topology_safe_pairs": auroc_on_topo_safe,
        "contamination_scenario": {
            "eval_clean": n_clean_eval, "hidden_injected": n_contam,
            "hidden_removed_by_flag": dropped_contam,
            "clean_wrongly_flagged": dropped_clean,
            "contamination_before": before, "contamination_after": after,
        },
    }
    os.makedirs("out", exist_ok=True)
    with open("out/manifold_flags_eval.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 68)
    print("Manifold flag validity — do flags catch hidden false negatives?")
    print("=" * 68)
    print(f"HuRI, leakage-free. hidden positives={len(hid_ok)}  clean negatives={len(cln_ok)}"
          f"  (abstained {abst:.0%})\n")
    print("1) Does the flag signal separate hidden positives from clean negatives?")
    print(f"     manifold resemblance  AUROC = {auroc_manifold:.4f}")
    print(f"     topology risk (base)  AUROC = {auroc_topology:.4f}")
    print(f"     flag lift (hidden ÷ clean flag-rate) = {flag_lift}×")
    print("\n2) Complementarity — on pairs topology calls SAFE, can manifold still")
    print(f"     find the hidden positives?  AUROC = {auroc_on_topo_safe}")
    c = report["contamination_scenario"]
    print(f"\n3) Eval-set cleanliness ({contamination:.0%} contamination):")
    print(f"     injected {c['hidden_injected']} hidden positives into {c['eval_clean']} clean;")
    print(f"     the flag removed {c['hidden_removed_by_flag']} of them "
          f"(contamination {c['contamination_before']} → {c['contamination_after']}),")
    print(f"     at the cost of flagging {c['clean_wrongly_flagged']} clean negatives.")
    print("\n(full report -> out/manifold_flags_eval.json)")
    return report


if __name__ == "__main__":
    main()
