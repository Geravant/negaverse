"""Does training an inductive (ESM2 sequence) PPI model on negaverse negatives
generalize better than random negatives — while keeping in-distribution accuracy?

This is the experiment behind Lucy's 12.07 framing (train on HuRI, predict novel
host-host PPIs) and the negaverse value question, done for an INDUCTIVE model —
one that scores a pair from the two protein SEQUENCES (ESM2), so it works on
proteins it never trained on. Transductive graph features can't do that.

Three negative-composition ARMS, all the same size, differing only in *how the
training negatives are chosen*:
  * random   — uniform veto-cleaned HuRI non-edges (the baseline).
  * stacked  — negaverse's shipped default (topology-hard tail re-ranked by fused
               biology confidence).
  * verified — stacked, then the LLM judge verifies the contested tail and every
               suspected_false_negative is DROPPED (no mislabeled positives).
               THIS is "llm-as-a-judge in set composing".

Three evaluation REGIMES, each graded against an INDEPENDENT gold yardstick
(never the negatives used to select — no grading with the selection ruler):
  * in_distribution   — random pair split; held-out HuRI positives vs Negatome gold.
                        Tests "accuracy maintained".
  * protein_disjoint  — proteins split into disjoint halves; train on one, test on
                        the other. No protein in both → true inductive generalization.
  * transfer_dryad    — train on HuRI, test on DRYAD's own labelled pos/neg (a
                        different lab/assay). ESM2 features are ID-agnostic, so this
                        is seamless. Cross-dataset generalization.

Yardsticks are gold (Negatome / DRYAD labels); features are ESM2 sequence, which is
independent of the topology/rule axis used to pick the training negatives.

    # compose once (runs the pipeline + LLM judge, cached), then evaluate:
    PYTHONPATH=. python3 scripts/bench_inductive_generalization.py
    #   --recompose         re-run the pipeline/judge (else reuse the cached records)
    #   --n-neg 3000        negatives per arm   --seeds 0 1 2   --models rf logreg
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from negaverse.cli import _load_dotenv
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space
from negaverse.io.embeddings import load_embeddings_npz
from negaverse.pipeline import PipelineConfig, run_pipeline
from negaverse.streams import build_filters, LiteratureFilter

HURI_EMB = "local-docs/huri/esm2_huri.npz"
DRYAD_EMB = "local-docs/dryad-ppi/esm2_t6_emb.npz"
DRYAD_TSV = "local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv"
SYMS = "local-docs/mappings/ensg_symbol.tsv"
CACHE = Path("out/inductive/huri_composed.jsonl")


# --- features -----------------------------------------------------------
def _feat(u, v, emb):
    """Order-invariant ESM2 pair feature (min|max concat) — the config that
    separated DRYAD pos/gold at AUROC ~0.93. None if either has no embedding."""
    a, b = emb.get(u), emb.get(v)
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.concatenate([np.minimum(a, b), np.maximum(a, b)])


def _matrix(pairs, emb):
    rows, kept = [], []
    for u, v in pairs:
        f = _feat(u, v, emb)
        if f is not None:
            rows.append(f); kept.append((u, v))
    return (np.asarray(rows) if rows else np.empty((0, 640))), kept


def _make_model(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(kind)


def _fit_eval(tr_pos, tr_neg, te_pos, te_neg, emb_tr, emb_te, models, seed, n_cap):
    """Train balanced (positives subsampled to the negative count) and score the
    held-out positives vs gold negatives. Returns {model: {auroc, auprc, n_*}}."""
    rng = np.random.default_rng(seed)
    Xtp, ptp = _matrix(tr_pos, emb_tr)
    Xtn, _ = _matrix(tr_neg, emb_tr)
    if len(Xtp) == 0 or len(Xtn) == 0:
        return {}
    n = min(len(Xtp), len(Xtn), n_cap)                 # balanced + capped
    Xtp = Xtp[rng.choice(len(Xtp), n, replace=False)]
    Xtn = Xtn[rng.choice(len(Xtn), n, replace=False)]
    Xtr = np.vstack([Xtp, Xtn]); ytr = np.r_[np.ones(n), np.zeros(n)]

    Xep, _ = _matrix(te_pos, emb_te)
    Xen, _ = _matrix(te_neg, emb_te)
    if len(Xep) == 0 or len(Xen) == 0:
        return {}
    Xte = np.vstack([Xep, Xen])
    yte = np.r_[np.ones(len(Xep)), np.zeros(len(Xen))]

    out = {}
    for kind in models:
        clf = _make_model(kind, seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        out[kind] = {"auroc": float(roc_auc_score(yte, p)),
                     "auprc": float(average_precision_score(yte, p)),
                     "n_train_each": n, "n_test_pos": len(Xep), "n_test_neg": len(Xen)}
    return out


# --- negative-set composition (pipeline + LLM judge, cached) -------------
def compose(graph, syms, n_train, seed, recompose):
    if CACHE.exists() and not recompose:
        recs = [json.loads(l) for l in CACHE.read_text().splitlines() if l.strip()]
        print(f"  reusing {len(recs)} cached composed records ({CACHE})")
        return recs
    print(f"  running pipeline (stacked, n_train={n_train}) + LLM judge on the contested tail ...")
    filters = build_filters("ppi", ["known_positive_veto", "structured", "topology", "rules"])
    filters.append(LiteratureFilter(enabled=True, provider="auto", votes=3, names=syms))
    cfg = PipelineConfig(modality="ppi", n_eval=200, n_train=n_train, max_pool=200_000,
                         seed=seed, train_selection="stacked",
                         false_negative_pct=0.05, gated_max=400)
    res = run_pipeline(graph, cfg, filters=filters)
    recs = [{"u": r.u, "v": r.v, "flags": list(r.flags), "confidence": r.confidence,
             "hardness": r.hardness} for r in res.records if r.mode == "train"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("".join(json.dumps(r) + "\n" for r in recs))
    jr = res.stats.get("risky_coverage", {})
    print(f"  composed {len(recs)} train negatives; judge coverage: {jr}")
    return recs


def _random_nonedges(graph, n, seed, emb):
    rng = np.random.default_rng(seed)
    nodes = [x for x in graph.g.nodes() if x in emb]      # embeddable only
    out, seen = [], set()
    tries, cap = 0, n * 80 + 5000
    while len(out) < n and tries < cap:
        tries += 1
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a == b or graph.g.has_edge(a, b) or k in seen:
            continue
        seen.add(k); out.append((a, b))
    return out


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=3000, help="negatives per arm (balanced train)")
    ap.add_argument("--n-train-pipeline", type=int, default=8000,
                    help="pipeline n_train to compose from (headroom for dropping FNs)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["rf", "logreg"], choices=["rf", "logreg"])
    ap.add_argument("--recompose", action="store_true")
    args = ap.parse_args()

    print("Loading HuRI + ESM2 + Negatome + DRYAD ...")
    g = load_huri_graph()
    huri_emb = load_embeddings_npz(HURI_EMB)
    syms = ({l.split("\t")[0]: l.split("\t")[1].strip()
             for l in Path(SYMS).read_text().splitlines() if "\t" in l} if Path(SYMS).exists() else {})
    pos_all = [tuple(e) for e in g.g.edges()]
    gold_huri = [tuple(p) for p in load_negatome_in_ensembl_space(set(g.g.nodes()))]

    # DRYAD (transfer target): its own embeddings + labelled pos/neg
    dryad_emb = load_embeddings_npz(DRYAD_EMB)
    d_pos, d_neg = [], []
    for line in Path(DRYAD_TSV).read_text().splitlines()[1:]:
        pr, cat = line.split("\t"); a, b = pr.split("_")
        (d_pos if cat == "positive" else d_neg).append((a, b))
    print(f"  HuRI: {len(pos_all)} pos, {len(gold_huri)} Negatome gold | "
          f"DRYAD: {len(d_pos)} pos, {len(d_neg)} gold | symbols: {len(syms)}")

    # ---- compose the three negative arms (llm-judge in the loop) ----
    print("Composing negative arms ...")
    composed = compose(g, syms, args.n_train_pipeline, seed=0, recompose=args.recompose)
    stacked = [(r["u"], r["v"]) for r in composed]
    verified = [(r["u"], r["v"]) for r in composed
                if "suspected_false_negative" not in r["flags"]]
    dropped = len(stacked) - len(verified)
    print(f"  stacked={len(stacked)}  verified={len(verified)}  "
          f"(judge/flags DROPPED {dropped} suspected false negatives)")

    def arm_negs(name, seed):
        if name == "random":
            return _random_nonedges(g, args.n_neg * 2, seed, huri_emb)
        return {"stacked": stacked, "verified": verified}[name]

    ARMS = ["random", "stacked", "verified"]

    # protein-disjoint split: hash each protein to side A/B (deterministic per seed)
    def side(p, seed):
        return int(hashlib.sha1(f"{seed}:{p}".encode()).hexdigest(), 16) & 1

    results = {}                      # (regime, arm, model) -> [auroc per seed]
    def _acc(regime, arm, model, m):
        results.setdefault((regime, arm, model), {"auroc": [], "auprc": []})
        results[(regime, arm, model)]["auroc"].append(m["auroc"])
        results[(regime, arm, model)]["auprc"].append(m["auprc"])

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # shuffle positives / gold once per seed
        pos = pos_all[:]; rng.shuffle(pos)
        gold = gold_huri[:]; rng.shuffle(gold)

        for arm in ARMS:
            negs = arm_negs(arm, seed)

            # (1) in_distribution — random 80/20 pair split, gold negatives for test
            cut = int(len(pos) * 0.2)
            te_pos, tr_pos = pos[:cut], pos[cut:]
            r = _fit_eval(tr_pos, negs, te_pos, gold, huri_emb, huri_emb,
                          args.models, seed, args.n_neg)
            for mdl, m in r.items():
                _acc("in_distribution", arm, mdl, m)

            # (2) protein_disjoint — train on side-A pairs, test on side-B pairs
            trA_pos = [(u, v) for u, v in pos if side(u, seed) == 0 and side(v, seed) == 0]
            teB_pos = [(u, v) for u, v in pos if side(u, seed) == 1 and side(v, seed) == 1]
            trA_neg = [(u, v) for u, v in negs if side(u, seed) == 0 and side(v, seed) == 0]
            teB_gold = [(u, v) for u, v in gold if side(u, seed) == 1 and side(v, seed) == 1]
            r = _fit_eval(trA_pos, trA_neg, teB_pos, teB_gold, huri_emb, huri_emb,
                          args.models, seed, args.n_neg)
            for mdl, m in r.items():
                _acc("protein_disjoint", arm, mdl, m)

            # (3) transfer_dryad — train all HuRI, test DRYAD gold pos/neg
            r = _fit_eval(pos, negs, d_pos, d_neg, huri_emb, dryad_emb,
                          args.models, seed, args.n_neg)
            for mdl, m in r.items():
                _acc("transfer_dryad", arm, mdl, m)
        print(f"  seed {seed} done")

    # ---- report ----
    mean = lambda regime, arm, mdl, k: (
        float(np.mean(results[(regime, arm, mdl)][k]))
        if (regime, arm, mdl) in results and results[(regime, arm, mdl)][k] else float("nan"))
    REGIMES = ["in_distribution", "protein_disjoint", "transfer_dryad"]
    RLABEL = {"in_distribution": "in-distribution (accuracy held)",
              "protein_disjoint": "protein-disjoint (generalization)",
              "transfer_dryad": "transfer→DRYAD (generalization)"}
    print("\n" + "=" * 78)
    print("INDUCTIVE GENERALIZATION — AUROC (mean over seeds), verified = llm-judge-cleaned")
    print("=" * 78)
    for mdl in args.models:
        print(f"\n  model = {mdl.upper()}")
        print(f"  {'regime':<34}{'random':>9}{'stacked':>9}{'verified':>9}   Δ(ver−rand)")
        for reg in REGIMES:
            rr = mean(reg, "random", mdl, "auroc")
            st = mean(reg, "stacked", mdl, "auroc")
            ve = mean(reg, "verified", mdl, "auroc")
            d = ve - rr
            print(f"  {RLABEL[reg]:<34}{rr:>9.3f}{st:>9.3f}{ve:>9.3f}   {d:>+8.3f}")

    out = {"config": vars(args),
           "composed": {"stacked": len(stacked), "verified": len(verified), "dropped_fn": dropped},
           "results": {f"{r}|{a}|{m}": {k: mean(r, a, m, k) for k in ("auroc", "auprc")}
                       for r in REGIMES for a in ARMS for m in args.models}}
    Path("out/inductive").mkdir(parents=True, exist_ok=True)
    Path("out/inductive/generalization.json").write_text(json.dumps(out, indent=2))
    print("\nwrote out/inductive/generalization.json")


if __name__ == "__main__":
    main()
