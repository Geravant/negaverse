"""Apply the inductive ESM2 PPI model to a novel candidate space and emit a
ranked shortlist of undocumented predictions — the input to AF2-Multimer triage.

Trains the same inductive model as bench_inductive_generalization.py (ESM2 concat +
RandomForest) on ALL HuRI positives + a chosen negative arm (default: the
llm-judge-verified set, reusing the composed-negatives cache), then scores every
admissible pair in the candidate space, SUBTRACTS everything documented in the
open PPI DBs (IntAct/BioGRID via rules/sources.yaml + the source graph's own
edges), and writes the top-K novel pairs.

Candidate spaces (prepared by scripts/prepare_*.py):
  * sars_host — host×host pairs among the SARS-CoV-2 interactome's human proteins
    (Lucy 12.07: predict novel HOST-host PPIs; the human-calibrated rules apply).
  * idg       — pairs among the IDG understudied kinome (Tdark+Tbio kinases).

    PYTHONPATH=. python3 scripts/predict_novel_candidates.py --space sars_host --top-k 50
    PYTHONPATH=. python3 scripts/predict_novel_candidates.py --space idg       --top-k 50
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from negaverse.io import load_huri_graph
from negaverse.io.embeddings import load_embeddings_npz
from negaverse.io.sources import load_positive_sources

HURI_EMB = "local-docs/huri/esm2_huri.npz"
COMPOSED = Path("out/inductive/huri_composed.jsonl")
SOURCES = "rules/sources.yaml"

SPACES = {
    "sars_host": {"emb": "local-docs/sars/esm2_sars.npz",
                  "meta": "local-docs/sars/proteins.tsv",       # node_id side source length
                  "seq": "local-docs/sars/sequences.tsv", "side": "host"},
    "idg": {"emb": "local-docs/idg/esm2_idg.npz",
            "meta": "local-docs/idg/kinases.tsv",               # uniprot symbol tdl length
            "seq": "local-docs/idg/sequences.tsv"},
}

# TOP-IDP per-residue disorder propensity (Campen 2008) — sequence-only, so it
# screens the model's disorder/sticky-hub failure mode BEFORE spending AF2 GPU
# time (validated: tracks AF2 disorder at Spearman 0.56; see af2_validate.py).
_TOP_IDP = {"A": 0.06, "R": 0.180, "N": 0.007, "D": 0.192, "C": 0.02, "Q": 0.318,
            "E": 0.736, "G": 0.166, "H": 0.303, "I": -0.486, "L": -0.326, "K": 0.586,
            "M": -0.397, "F": -0.697, "P": 0.987, "S": 0.341, "T": 0.059, "W": -0.884,
            "Y": -0.510, "V": -0.121}
_DISORDER_THR = 0.161            # HuRI 75th percentile — above this = disorder-risky


def _seq_disorder(seq):
    v = [_TOP_IDP.get(c, 0.0) for c in seq]
    return sum(v) / len(v) if v else 0.0


def _feat(u, v, emb):
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


def _train_model(neg_arm, seed):
    """Inductive model on ALL HuRI positives + the chosen negative arm. Negatives
    come from the composed cache (run bench_inductive_generalization.py first)."""
    if not COMPOSED.exists():
        raise SystemExit(f"missing {COMPOSED} — run bench_inductive_generalization.py first "
                         "to compose the negatives (pipeline + judge).")
    recs = [json.loads(l) for l in COMPOSED.read_text().splitlines() if l.strip()]
    if neg_arm == "verified":
        negs = [(r["u"], r["v"]) for r in recs if "suspected_false_negative" not in r["flags"]]
    else:
        negs = [(r["u"], r["v"]) for r in recs]
    g = load_huri_graph()
    emb = load_embeddings_npz(HURI_EMB)
    pos = [tuple(e) for e in g.g.edges()]
    Xp, _ = _matrix(pos, emb); Xn, _ = _matrix(negs, emb)
    rng = np.random.default_rng(seed)
    n = min(len(Xp), len(Xn))                       # balanced
    Xp = Xp[rng.choice(len(Xp), n, replace=False)]
    Xn = Xn[rng.choice(len(Xn), n, replace=False)]
    X = np.vstack([Xp, Xn]); y = np.r_[np.ones(n), np.zeros(n)]
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    print(f"  trained on {n} pos + {n} {neg_arm} neg (ESM2 concat + RF-300)")
    return clf


def _load_space(space):
    spec = SPACES[space]
    emb = load_embeddings_npz(spec["emb"])
    names, extra_meta = {}, {}
    proteins = []
    for line in Path(spec["meta"]).read_text().splitlines():
        if line.startswith(("node_id", "uniprot")) or not line.strip():
            continue
        f = line.split("\t")
        if space == "sars_host":
            nid, side = f[0], f[1]
            if side == spec["side"] and nid in emb:
                proteins.append(nid); names[nid] = nid
        else:                                        # idg
            acc, sym, tdl = f[0], f[1], f[2]
            if acc in emb:
                proteins.append(acc); names[acc] = sym; extra_meta[acc] = tdl
    return emb, sorted(set(proteins)), names, extra_meta


def _documented(proteins, extra_edges):
    """Union of open-DB positives among `proteins` (IntAct/BioGRID via sources.yaml,
    restricted to this protein set) plus any known edges passed in."""
    known, report = load_positive_sources(SOURCES, restrict_to=set(proteins))
    known = set(known) | set(extra_edges)
    print(f"  documented among candidates: {len(known)} pairs "
          f"(sources loaded: { {k: v for k, v in (report.get('loaded') or {}).items() if v} })")
    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=list(SPACES), required=True)
    ap.add_argument("--neg-arm", choices=["verified", "stacked"], default="verified")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-per-protein", type=int, default=3,
                    help="hub-diversity cap: max times any protein may appear in the "
                         "shortlist (an inductive model over-scores sticky hubs; 0 = no cap)")
    ap.add_argument("--max-disorder", type=float, default=_DISORDER_THR,
                    help="drop candidate pairs whose more-disordered protein exceeds this "
                         "TOP-IDP sequence-disorder score — the model's structural failure "
                         "mode, screened before AF2 (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Training inductive model ({args.neg_arm} negatives) ...")
    clf = _train_model(args.neg_arm, args.seed)

    print(f"Loading candidate space: {args.space} ...")
    emb, proteins, names, extra_meta = _load_space(args.space)
    print(f"  {len(proteins)} embedded candidate proteins → {len(proteins)*(len(proteins)-1)//2} pairs")

    # documented edges to subtract (novelty)
    extra_edges = set()
    if args.space == "sars_host":
        from negaverse.io import load_sars_cov2_graph
        sg = load_sars_cov2_graph()
        extra_edges = {frozenset(e) for e in sg.g.edges()}   # don't re-predict known SARS edges
    documented = _documented(proteins, extra_edges)

    # sequence disorder (TOP-IDP) per candidate protein — for the pre-AF2 screen
    sd = {}
    if SPACES[args.space].get("seq"):
        for l in open(SPACES[args.space]["seq"]):
            if "\t" in l:
                i, s = l.rstrip("\n").split("\t")[:2]; sd[i] = _seq_disorder(s)
    pair_dis = lambda u, v: max(sd.get(u, 0.0), sd.get(v, 0.0))

    # score every admissible, undocumented pair; optionally drop disorder-risky
    # pairs (the model over-scores disordered sticky proteins AF2 rejects anyway)
    all_cand = [(u, v) for u, v in combinations(proteins, 2)
                if frozenset((u, v)) not in documented]
    if args.max_disorder:
        cand = [pr for pr in all_cand if pair_dis(*pr) <= args.max_disorder]
        print(f"  disorder screen (TOP-IDP ≤ {args.max_disorder}): dropped "
              f"{len(all_cand)-len(cand)}/{len(all_cand)} disorder-risky pairs")
    else:
        cand = all_cand
    X, kept = _matrix(cand, emb)
    if len(X) == 0:
        raise SystemExit("no scorable candidate pairs")
    p = clf.predict_proba(X)[:, 1]
    order = np.argsort(-p)

    # An inductive sequence model over-scores "sticky" hub proteins, so the raw
    # top-K collapses onto one hub. Greedily pick down the ranking with a
    # per-protein cap so the shortlist is diverse and useful for AF2 triage.
    from collections import Counter
    raw_hub = Counter()
    for i in order[:args.top_k]:
        u, v = kept[i]; raw_hub[u] += 1; raw_hub[v] += 1
    cap, used, picked = args.max_per_protein, Counter(), []
    for i in order:
        u, v = kept[i]
        if cap and (used[u] >= cap or used[v] >= cap):
            continue
        picked.append(i); used[u] += 1; used[v] += 1
        if len(picked) >= args.top_k:
            break

    Path("out/inductive").mkdir(parents=True, exist_ok=True)
    out = Path(f"out/inductive/novel_{args.space}_{args.neg_arm}.tsv")
    with out.open("w") as fh:
        hdr = "rank\tprotein_a\tprotein_b\tname_a\tname_b\tp_interact\tmax_disorder"
        hdr += "\ttdl_a\ttdl_b\n" if args.space == "idg" else "\n"
        fh.write(hdr)
        for rank, i in enumerate(picked, 1):
            u, v = kept[i]
            row = f"{rank}\t{u}\t{v}\t{names.get(u,u)}\t{names.get(v,v)}\t{p[i]:.4f}\t{pair_dis(u,v):.3f}"
            row += f"\t{extra_meta.get(u,'')}\t{extra_meta.get(v,'')}\n" if args.space == "idg" else "\n"
            fh.write(row)

    top_hub, top_n = (raw_hub.most_common(1)[0] if raw_hub else ("-", 0))
    print(f"\nscored {len(kept)} novel pairs; wrote diverse top-{len(picked)} "
          f"(≤{cap}/protein) → {out}")
    print(f"  P(interact) range: {p.min():.3f}–{p.max():.3f}")
    print(f"  hub bias: without the cap, {names.get(top_hub,top_hub)} would occupy "
          f"{top_n}/{args.top_k} of the shortlist — the cap prevents a one-hub list.")
    print("\n  diversified top 10 (for AF2-Multimer triage):")
    for rank, i in enumerate(picked[:10], 1):
        u, v = kept[i]
        tag = f"  [{extra_meta.get(u,'')}/{extra_meta.get(v,'')}]" if args.space == "idg" else ""
        print(f"   {rank:>2}. {names.get(u,u)} × {names.get(v,v)}  p={p[i]:.3f}{tag}")


if __name__ == "__main__":
    main()
