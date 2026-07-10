"""PLI benchmark + 3D dashboard on PLINDER.

Hypothesis: pocket-fit-hard negatives (a ligand that FITS the pocket the way real
binders do, but isn't a documented binder) are better training negatives than
random protein-ligand pairs.

Benchmark (the PLI analog of the PPI circularity check): train a RandomForest
binder-classifier on PLINDER positives + (random | negaverse) negatives, test on
held-out positives + fixed random non-pairs, under two feature families:
  * "size"      — ligand heavy-atoms / MW / pocket residues / their ratio. This is
                  the space our pocket-fit selection lives in -> expect circular gain.
  * "chemistry" — ligand logP / TPSA / rings / QED. INDEPENDENT of size selection
                  -> the honest read on whether the negatives generalise.

Dashboard: out/plinder/report.html — a rotatable 3D map of protein-ligand pairs
(positive / random / negaverse) by pocket-fit likeness x ligand logP x ligand size.

    PYTHONPATH=. python scripts/pli_benchmark.py [--max-positives N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_plinder_graph
from negaverse.io.annotations import build_annotation_table
from negaverse.pipeline import PipelineConfig, run_pipeline
from negaverse.streams import PliPocketFitFilter

SEED = 0
FEATURES = {
    "size": ["_volume", "_mw", "_pocket", "_ratio"],
    "chemistry": ["logp", "tpsa", "rings", "qed"],
}


def _orient(graph, u, v):
    return (u, v) if graph.node_type.get(u) == "protein" else (v, u)


def _featurize(graph, ann, pairs, family):
    rows = []
    for u, v in pairs:
        prot, lig = _orient(graph, u, v)
        la, pa = ann.get(lig, {}), ann.get(prot, {})
        if family == "chemistry":
            rows.append([la.get(k, np.nan) for k in FEATURES["chemistry"]])
        else:
            vol, pv = la.get("volume"), pa.get("pocket_volume")
            rows.append([vol if vol is not None else np.nan,
                         la.get("mw", np.nan), pv if pv is not None else np.nan,
                         (vol / pv) if (vol and pv) else np.nan])
    return np.asarray(rows, dtype=float)


def _impute(train_X, *more):
    med = np.nanmedian(train_X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for X in (train_X, *more):
        Xi = np.where(np.isnan(X), med, X)
        out.append(Xi)
    return out


def _random_nonpairs(prots, ligs, pos_set, n, rng, exclude=None):
    exclude = exclude or set()
    out, seen = [], set()
    while len(out) < n:
        p, l = prots[rng.integers(len(prots))], ligs[rng.integers(len(ligs))]
        k = (p, l)
        if k not in pos_set and k not in exclude and k not in seen:
            seen.add(k); out.append((p, l))
    return out


def _negaverse_negs(train_pos, node_type, n, seed):
    tg = TypedInteractionGraph.from_edges(
        train_pos, node_type, admissible_types=[("protein", "ligand")], name="pli-train")
    cfg = PipelineConfig(modality="pli", n_eval=0, n_train=n, max_pool=50_000, seed=seed,
                         match_on_type="protein",
                         filters=["known_positive_veto", "structured", "rules", "pli_pocket_fit"])
    res = run_pipeline(tg, cfg)
    return [(r.u, r.v) for r in res.records if r.mode == "train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positives", type=int, default=20_000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading PLINDER ...")
    graph = load_plinder_graph()
    ann = build_annotation_table()
    print("  graph:", graph.summary())

    pos = [tuple(e) for e in graph.g.edges()]
    pos = [_orient(graph, u, v) for u, v in pos]                # (protein, ligand)
    rng.shuffle(pos)
    if args.max_positives and len(pos) > args.max_positives:
        pos = pos[:args.max_positives]
    n_test = int(len(pos) * args.test_frac)
    test_pos, train_pos = pos[:n_test], pos[n_test:]
    pos_set = set(pos)
    prots = graph.nodes_of_type("protein")
    ligs = graph.nodes_of_type("ligand")
    node_type = dict(graph.node_type)

    # fixed unbiased random TEST negatives (same for every strategy)
    test_neg = _random_nonpairs(prots, ligs, pos_set, len(test_pos), rng)
    test_excl = set(test_neg)

    # training negatives per strategy
    print("Generating training negatives ...")
    strat_negs = {
        "random": _random_nonpairs(prots, ligs, pos_set, len(train_pos), rng, exclude=test_excl),
        "negaverse": _negaverse_negs(train_pos, node_type, len(train_pos), args.seed),
    }
    for k, v in strat_negs.items():
        print(f"  {k}: {len(v)} negatives")

    print("\n=== PLI benchmark (train on positives + random|negaverse negatives; "
          "test on held-out positives + random non-pairs) ===")
    results = {}
    for family in FEATURES:
        Xte = _featurize(graph, ann, list(test_pos) + list(test_neg), family)
        yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]
        line = [f"  [{family:9}]"]
        for strat, negs in strat_negs.items():
            Xtr = _featurize(graph, ann, list(train_pos) + list(negs), family)
            ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(negs))]
            Xtr, Xte_i = _impute(Xtr, Xte)
            clf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte_i)[:, 1]
            au = roc_auc_score(yte, p); ap_ = average_precision_score(yte, p)
            results[(family, strat)] = au
            line.append(f"{strat} AUROC={au:.4f} AUPRC={ap_:.4f}")
        r, nv = results.get((family, "random")), results.get((family, "negaverse"))
        line.append(f"| Δ={nv - r:+.4f}")
        print("   ".join(line))

    _dashboard(graph, ann, test_pos, strat_negs["random"], strat_negs["negaverse"], results)


def _dashboard(graph, ann, pos, rand, nv, results):
    from negaverse.viz.bench3d import render_3d_report
    fit = PliPocketFitFilter(); fit.fit(graph)

    def pts(pairs):
        out = []
        for u, v in pairs:
            prot, lig = _orient(graph, u, v)
            la = ann.get(lig, {})
            sc = fit.score(graph, prot, lig)
            like = (sc.evidence or {}).get("pocket_fit_likeness")
            lp, vol = la.get("logp"), la.get("volume")
            if like is None or lp is None or vol is None:
                continue
            out.append([like, lp, float(vol)])
        return np.asarray(out, dtype=float)

    classes = [{"name": "positive (real binder)", "color": "#2a9d8f", "points": pts(pos)},
               {"name": "random non-pair", "color": "#adb5bd", "points": pts(rand)},
               {"name": "negaverse (pocket-fit hard)", "color": "#7b2ff7", "points": pts(nv)}]
    rep = render_3d_report(
        Path("out") / "plinder" / "report.html",
        title="PLINDER — protein–ligand negatives (PLI)",
        subtitle="Bipartite protein(UniProt)–ligand(CCD) graph · pocket-fit hardness",
        classes=classes, axis_labels=("pocket-fit likeness", "ligand logP", "ligand heavy atoms"),
        summary_rows=[(f"{fam} features · {strat}", f"AUROC {au:.3f}")
                      for (fam, strat), au in results.items()],
        caption="Each point is a protein–ligand PAIR. x = pocket-fit likeness (how typical the "
                "ligand's size is for that pocket, the hardness signal); y = ligand logP; z = ligand "
                "heavy-atom count. Teal = real binders, grey = random non-pairs, "
                "<b>purple = negaverse's pocket-fit-hard negatives</b> — selected to sit at high "
                "pocket-fit likeness (x), i.e. they LOOK like binders by size, unlike random pairs. "
                "The benchmark table shows whether that hardness helps under size features (aligned "
                "with selection) vs independent chemistry features (the honest read).")
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
