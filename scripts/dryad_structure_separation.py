"""Protein-STRUCTURE-based separation on the DRYAD benchmark
(doi:10.5061/dryad.15dv41p84, "A dataset for predicting protein-protein interactions").

The DRYAD topology experiment doesn't apply — that dataset is sequence/structure,
not a network. Per the locked spec, ESM2 embeddings are our structure/viz signal.
So the analog of the UPNA topology test is: do ESM2 sequence embeddings separate
the benchmark's positive pairs from its negatives?

Pipeline (each stage cached under local-docs/dryad-ppi/):
  1. balanced subsample of positives_and_negatives.tsv (UniProt pairs + labels)
  2. fetch sequences from UniProt REST
  3. embed each protein with ESM2-t6-8M (mean-pooled), via transformers
  4. separation, reported three honest ways:
       * unsupervised   — cosine similarity of the two embeddings (no training)
       * CV (standard)  — 5-fold on Hadamard pair features (protein overlap => optimistic)
       * protein-disjoint — train/test share NO proteins (the fair "unseen proteins" test)
  5. PCA-2D scatter of pair embeddings, coloured by label -> out/

    PYTHONPATH=. python scripts/dryad_structure_separation.py
"""
from __future__ import annotations

import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

DIR = Path("local-docs/dryad-ppi")
BENCH = DIR / "benchmarks/benchmarks/positives_and_negatives.tsv"
SEQ_CACHE = DIR / "sequences.tsv"
EMB_CACHE = DIR / "esm2_t6_emb.npz"
MODEL = "facebook/esm2_t6_8M_UR50D"
SEED = 0
N_PER_CLASS = 1200
MAX_LEN = 1022


# --- 1. subsample -------------------------------------------------------
def load_subsample():
    rng = np.random.default_rng(SEED)
    pos, neg = [], []
    with open(BENCH) as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for pair, cat in r:
            (pos if cat == "positive" else neg).append(tuple(pair.split("_")))
    def take(lst):
        idx = rng.choice(len(lst), size=min(N_PER_CLASS, len(lst)), replace=False)
        return [lst[i] for i in idx]
    pos, neg = take(pos), take(neg)
    prots = {x for p in pos + neg for x in p}
    return pos, neg, sorted(prots)


# --- 2. sequences -------------------------------------------------------
def fetch_sequences(accessions):
    cache = {}
    if SEQ_CACHE.exists():
        with open(SEQ_CACHE) as f:
            for line in f:
                a, s = line.rstrip("\n").split("\t")
                cache[a] = s
    missing = [a for a in accessions if a not in cache]
    if missing:
        print(f"  fetching {len(missing)} sequences from UniProt ...")
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            q = urllib.parse.urlencode({"accessions": ",".join(chunk),
                                        "fields": "accession,sequence", "format": "json"})
            url = f"https://rest.uniprot.org/uniprotkb/accessions?{q}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url), timeout=30, context=_SSL) as fh:
                        data = json.load(fh)
                    for e in data.get("results", []):
                        acc = e.get("primaryAccession")
                        seq = e.get("sequence", {}).get("value")
                        if acc and seq:
                            cache[acc] = seq
                    break
                except Exception as ex:
                    print(f"    chunk {i//100}: retry {attempt+1} ({ex})")
                    time.sleep(2 * (attempt + 1))
        SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(SEQ_CACHE, "w") as f:
            for a in sorted(cache):
                f.write(f"{a}\t{cache[a]}\n")
    return cache


# --- 3. ESM2 embeddings -------------------------------------------------
def embed(proteins, seqs):
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE, allow_pickle=True)
        cached = {k: v for k, v in zip(d["ids"], d["emb"])}
    else:
        cached = {}
    todo = [p for p in proteins if p in seqs and p not in cached]
    if todo:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"  embedding {len(todo)} proteins with {MODEL} (CPU) ...")
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).eval()
        B = 16
        with torch.no_grad():
            for i in range(0, len(todo), B):
                batch = todo[i:i + B]
                enc = tok([seqs[p][:MAX_LEN] for p in batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_LEN)
                out = model(**enc).last_hidden_state           # (B, L, 320)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1)      # mean over real tokens
                for p, v in zip(batch, pooled.cpu().numpy()):
                    cached[p] = v.astype(np.float32)
                if (i // B) % 20 == 0:
                    print(f"    {i+len(batch)}/{len(todo)}")
        ids = list(cached); emb = np.stack([cached[k] for k in ids])
        np.savez(EMB_CACHE, ids=np.array(ids), emb=emb)
    return cached


# --- 4/5. separation + plot --------------------------------------------
def _cosine(emb, pairs):
    out = []
    for a, b in pairs:
        u, v = emb.get(a), emb.get(b)
        if u is None or v is None:
            out.append(np.nan); continue
        out.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    return np.array(out)


def _hadamard(emb, pairs):
    X, keep = [], []
    for k, (a, b) in enumerate(pairs):
        u, v = emb.get(a), emb.get(b)
        if u is None or v is None:
            continue
        X.append(u * v); keep.append(k)
    return np.asarray(X), keep


def main():
    print("1. subsampling DRYAD benchmark ...")
    pos, neg, prots = load_subsample()
    print(f"   {len(pos)} pos + {len(neg)} neg; {len(prots)} unique proteins")
    print("2. sequences ...")
    seqs = fetch_sequences(prots)
    print(f"   have sequences for {len(seqs)}/{len(prots)}")
    print("3. ESM2 embeddings ...")
    emb = embed(prots, seqs)
    print(f"   embedded {len(emb)} proteins (dim {next(iter(emb.values())).shape[0]})")

    pairs = pos + neg
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    # unsupervised: cosine similarity
    cos = _cosine(emb, pairs)
    m = ~np.isnan(cos)
    au_cos = roc_auc_score(y[m], cos[m])

    # supervised: Hadamard features
    X, keep = _hadamard(emb, pairs)
    yk = y[keep]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = []
    for tr, te in skf.split(X, yk):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], yk[tr])
        cv.append(roc_auc_score(yk[te], clf.predict_proba(X[te])[:, 1]))
    au_cv = float(np.mean(cv))

    # protein-disjoint split (fair: train & test share no proteins)
    au_disj = _protein_disjoint(emb, pos, neg)

    print("\n=== structure (ESM2) separation on DRYAD: AUROC(pos vs neg) ===")
    print(f"  unsupervised cosine        AUROC={au_cos:.3f}   (no training, clean)")
    print(f"  supervised 5-fold CV       AUROC={au_cv:.3f}   (protein overlap -> optimistic)")
    print(f"  supervised protein-disjoint AUROC={au_disj:.3f}   (unseen proteins -> fair)")

    # negaverse's OWN negatives: build a graph from DRYAD's positive pairs, run the
    # pipeline to pick topology-hard negatives, and place them in the SAME ESM2 space.
    # (DRYAD's positive graph is sparse, so this is a weak topology signal — the point
    #  is to see where our negatives land structurally, not to claim they're hard here.)
    nvX = None
    try:
        from negaverse.graph import TypedInteractionGraph
        from negaverse.pipeline import PipelineConfig, run_pipeline
        tg = TypedInteractionGraph.from_edges(
            [tuple(p) for p in pos], {p: "protein" for p in prots},
            admissible_types=[("protein", "protein")], name="dryad-pos")
        cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=len(neg), max_pool=20000,
                             seed=SEED, filters=["structured", "topology"])
        res = run_pipeline(tg, cfg)
        nv_pairs = [(r.u, r.v) for r in res.records if r.mode == "train"]
        nvX, _ = _hadamard(emb, nv_pairs)
        print(f"  negaverse negatives (from DRYAD positive graph): {len(nvX)} placed in ESM2 space")
    except Exception as e:
        print(f"  (skipped negaverse overlay: {e})")
    _plot(cos, y, m, X, yk, {"cosine": au_cos, "cv": au_cv, "disjoint": au_disj}, nvX=nvX)


def _protein_disjoint(emb, pos, neg):
    rng = np.random.default_rng(SEED)
    prots = sorted({x for p in pos + neg for x in p})
    rng.shuffle(prots)
    cut = int(len(prots) * 0.6)
    train_p, test_p = set(prots[:cut]), set(prots[cut:])
    def split(pairs, S):
        return [p for p in pairs if p[0] in S and p[1] in S]
    trX, trY = _hadamard(emb, split(pos, train_p) + split(neg, train_p))[0], None
    tr_pairs = split(pos, train_p) + split(neg, train_p)
    te_pairs = split(pos, test_p) + split(neg, test_p)
    trX, trk = _hadamard(emb, tr_pairs)
    teX, tek = _hadamard(emb, te_pairs)
    trY = np.r_[np.ones(len(split(pos, train_p))), np.zeros(len(split(neg, train_p)))][trk]
    teY = np.r_[np.ones(len(split(pos, test_p))), np.zeros(len(split(neg, test_p)))][tek]
    if len(set(teY)) < 2 or len(set(trY)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000).fit(trX, trY)
    return roc_auc_score(teY, clf.predict_proba(teX)[:, 1])


def _plot(cos, y, m, X, yk, aurocs, nvX=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    out = Path("out"); out.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(-1, 1, 41)
    ax1.hist(cos[m & (y == 1)], bins=bins, density=True, histtype="step", lw=2,
             color="#2a9d8f", label="positive")
    ax1.hist(cos[m & (y == 0)], bins=bins, density=True, histtype="step", lw=2,
             color="#e76f51", label="negative")
    ax1.set_title(f"ESM2 cosine similarity  (AUROC={aurocs['cosine']:.3f})")
    ax1.set_xlabel("cosine(emb_u, emb_v)"); ax1.set_ylabel("density"); ax1.legend()

    p2 = PCA(2, random_state=SEED).fit_transform(X)
    for lab, col, name in [(1, "#2a9d8f", "positive"), (0, "#e76f51", "negative")]:
        s = yk == lab
        ax2.scatter(p2[s, 0], p2[s, 1], s=6, alpha=0.4, color=col, label=name)
    ax2.set_title("PCA of ESM2 Hadamard pair-embeddings")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.legend()
    fig.suptitle(f"DRYAD structure (ESM2) separation — CV AUROC={aurocs['cv']:.3f}, "
                 f"protein-disjoint={aurocs['disjoint']:.3f}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "dryad_structure_separation.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")

    # interactive 3D report — PCA-3 of the ESM2 pair-embeddings, coloured by label
    from negaverse.viz.bench3d import render_3d_report
    pca3 = PCA(3, random_state=SEED).fit(X)
    p3 = pca3.transform(X)
    classes = [{"name": "positive (interacting)", "color": "#2a9d8f", "points": p3[yk == 1]},
               {"name": "negative (non-interacting)", "color": "#e63946", "points": p3[yk == 0]}]
    if nvX is not None and len(nvX):
        classes.append({"name": "negaverse (our hard negative)", "color": "#7b2ff7",
                        "points": pca3.transform(nvX)})
    rep = render_3d_report(
        out / "dryad" / "report.html",
        title="DRYAD — protein structure (ESM2) separation",
        subtitle="Compound benchmark pair-list · ESM2-t6 mean-pooled embeddings · "
                 "Hadamard pair vector",
        classes=classes, axis_labels=("ESM2 PC1", "ESM2 PC2", "ESM2 PC3"),
        summary_rows=[("ESM2 cosine (unsupervised)", f"AUROC {aurocs['cosine']:.3f}"),
                      ("supervised 5-fold CV", f"AUROC {aurocs['cv']:.3f}"),
                      ("supervised protein-disjoint (fair)", f"AUROC {aurocs['disjoint']:.3f}")],
        caption="Each point is a protein PAIR, placed by the top-3 principal components of its "
                "ESM2 Hadamard embedding. Teal = real interactions, red = DRYAD's non-interactions; "
                "the two overlap heavily (structure alone separates at ~0.72 cosine / ~0.89 "
                "supervised). <b>Purple = negaverse's own negatives</b>, generated by running the "
                "pipeline on DRYAD's positive pairs and projected into this same ESM2 space. DRYAD "
                "is a labelled pair-list with a sparse positive graph, so our topology selection is "
                "weak here — the honest read is where our negatives LAND structurally (they sit "
                "inside the same cloud), not a claim of hardness. The structure signal itself is "
                "what generalises (protein-disjoint 0.885), which is why ESM2 belongs as a "
                "supervised feature — see docs/BENCHMARK-FINDINGS.md F-4.")
    print(f"wrote {rep}")


if __name__ == "__main__":
    main()
