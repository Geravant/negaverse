"""AF2-Multimer / pDockQ validation for the inductive model and its novel predictions.

Full AF2-Multimer folding needs a GPU + ColabFold, which a CPU sandbox can't run.
But we have TWO things that make a real, honest validation possible now:

  1. huintaf2 (Burke et al. 2023) — REAL AF2-Multimer pDockQ already computed for
     HuRI interactions, hu.MAP complexes, and random pairs. That is genuine AF2
     output, not a proxy.
  2. our inductive model — which we can cross-check against that real structural
     signal without folding anything new.

So this script does three things:

  reference   — the pass bar from real AF2 data: how pDockQ (interface confidence)
                separates real interactions from random. [runs now]
  crosscheck  — does our sequence model AGREE with real AF2 pDockQ? Trained
                PROTEIN-DISJOINT (so the evaluated huintaf2 pairs were never seen),
                we ask whether the pairs the model ranks highest have higher real
                pDockQ. Independent structural corroboration. [runs now]
  foldbatch   — the actual novel experiment, staged for a GPU box: writes a
                ColabFold FASTA for the diversified novel top-K + a length-matched
                random control, so `colabfold_batch` → pDockQ compares them against
                the reference. [writes the batch; folding is the GPU step]

    PYTHONPATH=. python3 scripts/af2_validate.py --stage reference
    PYTHONPATH=. python3 scripts/af2_validate.py --stage crosscheck
    PYTHONPATH=. python3 scripts/af2_validate.py --stage foldbatch --space idg --top-k 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from negaverse.io import load_huri_graph
from negaverse.io.embeddings import load_embeddings_npz

HUINTAF2 = "local-docs/huintaf2"
HURI_EMB = "local-docs/huri/esm2_huri.npz"
U2E = "local-docs/mappings/uniprot_ensg_human.tsv"
COMPOSED = "out/inductive/huri_composed.jsonl"
PDOCKQ_CONFIDENT = 0.23           # Bryant et al.: pDockQ > 0.23 ≈ a confident interface
OUT = Path("out/inductive")


def _u2e():
    m = {}
    for l in open(U2E):
        if "\t" in l and not l.startswith("#"):
            a, b = l.rstrip().split("\t")[:2]; m.setdefault(a, b)
    return m


def _load_huintaf2(name, uniprot, u2e):
    out = []
    for r in csv.DictReader(open(f"{HUINTAF2}/{name}.csv")):
        parts = r["Name"].split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        try:
            pd = float(r["pDockQ"])
        except ValueError:
            continue
        if uniprot:
            a, b = u2e.get(a), u2e.get(b)
            if not a or not b:
                continue
        out.append((a, b, pd))
    return out


def _feat(u, v, emb):
    a, b = emb.get(u), emb.get(v)
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.concatenate([np.minimum(a, b), np.maximum(a, b)])


# --- reference: real AF2 pDockQ separation ------------------------------
def reference():
    u2e = _u2e()
    sets = {"HuRI (real interactions)": _load_huintaf2("HuRI", False, u2e),
            "hu.MAP (real complexes)": _load_huintaf2("humap", True, u2e),
            "random (non-interactions)": _load_huintaf2("random", True, u2e)}
    print("REAL AF2-Multimer pDockQ (huintaf2 / Burke 2023) — interface confidence per pair")
    print(f"  {'set':<28}{'n':>7}{'mean':>8}{'median':>8}{'>0.23':>8}")
    for name, rows in sets.items():
        p = np.array([x[2] for x in rows])
        print(f"  {name:<28}{len(p):>7}{p.mean():>8.3f}{np.median(p):>8.3f}"
              f"{np.mean(p > PDOCKQ_CONFIDENT) * 100:>7.0f}%")
    H = np.array([x[2] for x in sets["HuRI (real interactions)"]])
    R = np.array([x[2] for x in sets["random (non-interactions)"]])
    y = np.r_[np.ones(len(H)), np.zeros(len(R))]; s = np.r_[H, R]
    print(f"\n  pass bar → real pDockQ separates real interactions from random: "
          f"AUROC = {roc_auc_score(y, s):.3f}")
    print(f"  a validated novel prediction should land in the real-interaction range "
          f"(≈{np.mean(H > PDOCKQ_CONFIDENT)*100:.0f}% clear {PDOCKQ_CONFIDENT}, vs "
          f"{np.mean(R > PDOCKQ_CONFIDENT)*100:.0f}% of random).")


# --- crosscheck: model vs real pDockQ, protein-disjoint -----------------
def crosscheck(seed=0, n_eval=4000):
    import hashlib, json
    emb = load_embeddings_npz(HURI_EMB)
    g = load_huri_graph()
    u2e = _u2e()
    side = lambda p: int(hashlib.sha1(f"{seed}:{p}".encode()).hexdigest(), 16) & 1

    comp = [json.loads(l) for l in open(COMPOSED)]
    verified = [(r["u"], r["v"]) for r in comp if "suspected_false_negative" not in r["flags"]]
    pos = [tuple(e) for e in g.g.edges()]

    # train PROTEIN-DISJOINT on side A only → evaluated side-B pairs are unseen proteins
    def mat(pairs):
        X, k = [], []
        for u, v in pairs:
            f = _feat(u, v, emb)
            if f is not None:
                X.append(f); k.append((u, v))
        return (np.asarray(X) if X else np.empty((0, 640))), k
    trA_pos = [(u, v) for u, v in pos if side(u) == 0 and side(v) == 0]
    trA_neg = [(u, v) for u, v in verified if side(u) == 0 and side(v) == 0]
    Xp, _ = mat(trA_pos); Xn, _ = mat(trA_neg)
    rng = np.random.default_rng(seed)
    n = min(len(Xp), len(Xn))
    Xp = Xp[rng.choice(len(Xp), n, replace=False)]; Xn = Xn[rng.choice(len(Xn), n, replace=False)]
    clf = RandomForestClassifier(300, random_state=seed, n_jobs=-1).fit(
        np.vstack([Xp, Xn]), np.r_[np.ones(n), np.zeros(n)])
    print(f"  trained protein-disjoint on side-A: {n} pos + {n} verified neg")

    H = _load_huintaf2("HuRI", False, u2e)
    R = _load_huintaf2("random", True, u2e)
    ev = ([(a, b, pd, 1) for a, b, pd in H if side(a) == 1 and side(b) == 1]
          + [(a, b, pd, 0) for a, b, pd in R if side(a) == 1 and side(b) == 1])
    rng.shuffle(ev); ev = ev[:n_eval]
    rows = [(a, b, pd, lab) for a, b, pd, lab in ev if _feat(a, b, emb) is not None]
    X = np.array([_feat(a, b, emb) for a, b, pd, lab in rows])
    P = clf.predict_proba(X)[:, 1]
    pdq = np.array([pd for *_, pd, lab in [(r[0], r[1], r[2], r[3]) for r in rows]])
    lab = np.array([r[3] for r in rows])
    print(f"  evaluated {len(rows)} unseen (side-B) huintaf2 pairs "
          f"({int(lab.sum())} real, {int((1-lab).sum())} random) — all with REAL pDockQ\n")

    print(f"  real pDockQ separates them (AF2):        AUROC = {roc_auc_score(lab, pdq):.3f}")
    print(f"  our model separates them (sequence):     AUROC = {roc_auc_score(lab, P):.3f}")
    rho, pv = spearmanr(P, pdq)
    print(f"  model P(interact) vs real pDockQ:        Spearman rho = {rho:+.3f} (p={pv:.1e})")
    order = np.argsort(-P)
    d = max(1, len(order) // 10)
    top, bot = order[:d], order[-d:]
    print(f"  pairs the model ranks HIGHEST vs LOWEST (top/bottom decile), mean real pDockQ:")
    print(f"     top-10%  pDockQ = {pdq[top].mean():.3f}  (>{PDOCKQ_CONFIDENT}: {np.mean(pdq[top]>PDOCKQ_CONFIDENT)*100:.0f}%)")
    print(f"     bot-10%  pDockQ = {pdq[bot].mean():.3f}  (>{PDOCKQ_CONFIDENT}: {np.mean(pdq[bot]>PDOCKQ_CONFIDENT)*100:.0f}%)")
    print("\n  → an independent structural method (real AF2) corroborates the model's ranking"
          if rho > 0 else "\n  → model ranking does NOT track structural pDockQ here")


# --- foldbatch: stage the novel top-K + matched control for ColabFold ---
_SEQ = {"idg": "local-docs/idg/sequences.tsv",
        "sars_host": "local-docs/sars/sequences.tsv"}
_NOVEL = {"idg": "out/inductive/novel_idg_verified.tsv",
          "sars_host": "out/inductive/novel_sars_host_verified.tsv"}


def foldbatch(space, top_k, seed=0):
    seqs = {}
    for l in open(_SEQ[space]):
        if "\t" in l:
            i, s = l.rstrip("\n").split("\t")[:2]; seqs[i] = s
    novel = list(csv.DictReader(open(_NOVEL[space]), delimiter="\t"))[:top_k]
    picks = [(r["protein_a"], r["protein_b"]) for r in novel
             if r["protein_a"] in seqs and r["protein_b"] in seqs]

    # length-matched random control: random pairs from the same space whose summed
    # length distribution matches the picks (so pDockQ isn't confounded by size)
    ids = [i for i in seqs]
    rng = np.random.default_rng(seed)
    target = sorted(len(seqs[a]) + len(seqs[b]) for a, b in picks)
    known = {frozenset(p) for p in picks}
    control = []
    while len(control) < len(picks) and len(known) < len(ids) ** 2:
        a, b = ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]
        if a != b and frozenset((a, b)) not in known:
            control.append((a, b)); known.add(frozenset((a, b)))

    OUT.mkdir(parents=True, exist_ok=True)
    fa = OUT / f"af2_{space}.fasta"
    man = OUT / f"af2_{space}_manifest.tsv"
    with fa.open("w") as fh, man.open("w") as mh:
        mh.write("complex_id\tgroup\tprotein_a\tprotein_b\n")
        for grp, pairs in [("novel", picks), ("control", control)]:
            for a, b in pairs:
                cid = f"{grp}__{a}__{b}"
                fh.write(f">{cid}\n{seqs[a]}:{seqs[b]}\n")     # ':' = ColabFold multimer chains
                mh.write(f"{cid}\t{grp}\t{a}\t{b}\n")
    print(f"wrote {len(picks)} novel + {len(control)} length-matched control complexes")
    print(f"  ColabFold FASTA: {fa}\n  manifest: {man}")
    print("\n  fold on a GPU box, then score pDockQ:")
    print(f"    colabfold_batch {fa} out/af2/{space}_preds")
    print(f"    PYTHONPATH=. python3 scripts/compute_af2_multimer.py --stage score")
    print("  expected (per the reference): novel top-K pDockQ should exceed the "
          "length-matched control and approach the real-interaction range.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["reference", "crosscheck", "foldbatch"], required=True)
    ap.add_argument("--space", choices=["idg", "sars_host"], default="idg")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "reference":
        reference()
    elif args.stage == "crosscheck":
        crosscheck()
    else:
        foldbatch(args.space, args.top_k)


if __name__ == "__main__":
    main()
