"""Repeat the inductive-generalization arm comparison with D-SCRIPT — the real,
published inductive architecture Lucy named (12.07) — instead of the ESM2+RF
prototype. Does a structure-aware sequence model reach the same verdict: negaverse
`stacked`/`verified` negatives generalize better than `random`?

D-SCRIPT (Sledzieski 2021) scores a pair from the two sequences via per-residue
Bepler-Berger embeddings + a contact-map CNN. Those per-residue embeddings are
~8 MB/protein, and training is CPU-bound here, so this runs at a BOUNDED scale
(capped sequence length + capped pairs) — enough to check whether the arm ordering
reproduces, not to beat the RF numbers on absolute AUROC.

Arms: random / stacked / verified (verified = judge-cleaned, from the composed
cache written by bench_inductive_generalization.py). Regimes: in_distribution and
protein_disjoint, graded against Negatome gold (same discipline as the RF bench).

    PYTHONPATH=. python3 scripts/bench_dscript.py            # bounded real run
    #   --n 500  --epochs 6  --max-len 500  --regimes protein_disjoint in_distribution
Requires the isolated venv: .venv-dscript (see session notes) with the lm_v1 model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space

DSCRIPT = ".venv-dscript/bin/dscript"
DEV = "-1"                                   # CPU (D-SCRIPT device is a GPU int; -1 = CPU)
SEQS = "local-docs/huri/sequences_ensg.tsv"
COMPOSED = "out/inductive/huri_composed.jsonl"
WORK = Path("out/dscript")


def _run(cmd):
    print("   $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_seqs(max_len):
    seq = {}
    for line in open(SEQS):
        if "\t" not in line:
            continue
        i, s = line.rstrip("\n").split("\t")[:2]
        if 30 <= len(s):
            seq[i] = s[:max_len]                # truncate long proteins (embedding/contact cost)
    return seq


def _write_pairs(path, pairs, labels):
    with open(path, "w") as fh:
        for (u, v), y in zip(pairs, labels):
            fh.write(f"{u}\t{v}\t{y}\n")


def _predict_auroc(name, test_pairs, labels, model, emb):
    pred = WORK / f"{name}_pred.tsv"
    _write_pairs(WORK / f"{name}_topred.tsv", test_pairs, [0] * len(test_pairs))
    _run([DSCRIPT, "predict", "--pairs", str(WORK / f"{name}_topred.tsv"),
          "--model", model, "--embeddings", emb, "-d", DEV, "--outfile", str(pred)])
    score = {}
    for line in open(pred):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            try:
                score[(p[0], p[1])] = float(p[2])
            except ValueError:
                pass
    s, y = [], []
    for (u, v), lab in zip(test_pairs, labels):
        if (u, v) in score:
            s.append(score[(u, v)]); y.append(lab)
        elif (v, u) in score:
            s.append(score[(v, u)]); y.append(lab)
    if len(set(y)) < 2:
        return float("nan"), len(y)
    return float(roc_auc_score(y, s)), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="pos and neg per arm (train)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regimes", nargs="+", default=["protein_disjoint", "in_distribution"])
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seq = _load_seqs(args.max_len)
    g = load_huri_graph()
    have = lambda u, v: u in seq and v in seq
    pos_all = [(u, v) for u, v in g.g.edges() if have(u, v)]
    gold_all = [(a, b) for a, b in load_negatome_in_ensembl_space(set(g.g.nodes())) if have(a, b)]
    comp = [json.loads(l) for l in open(COMPOSED)]
    stacked = [(r["u"], r["v"]) for r in comp if have(r["u"], r["v"])]
    verified = [(r["u"], r["v"]) for r in comp
                if "suspected_false_negative" not in r["flags"] and have(r["u"], r["v"])]

    def random_negs(k):
        nodes = list(seq); out, seen = [], set()
        while len(out) < k and len(seen) < k * 60 + 2000:
            a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
            key = frozenset((a, b)); seen.add(key)
            if a != b and not g.g.has_edge(a, b) and key not in {frozenset(p) for p in out}:
                out.append((a, b))
        return out

    arms = {"random": random_negs(args.n * 3), "stacked": stacked, "verified": verified}
    print(f"pool: {len(pos_all)} pos(seq), gold {len(gold_all)}, "
          f"stacked {len(stacked)}, verified {len(verified)}")

    def side(p):
        return int(hashlib.sha1(f"{args.seed}:{p}".encode()).hexdigest(), 16) & 1

    # collect all pairs we'll use, embed their protein union ONCE
    pos = pos_all[:]; rng.shuffle(pos)
    gold = gold_all[:]; rng.shuffle(gold)
    used_pairs = list(pos[:args.n * 3]) + list(gold)
    for a in arms.values():
        used_pairs += list(a[:args.n * 3])
    universe = sorted({p for pr in used_pairs for p in pr})
    fasta = WORK / "universe.fasta"
    with fasta.open("w") as fh:
        for p in universe:
            fh.write(f">{p}\n{seq[p]}\n")
    emb = str(WORK / "universe.h5")
    print(f"embedding {len(universe)} proteins (Bepler-Berger, ~8MB each) ...")
    if not Path(emb).exists():
        _run([DSCRIPT, "embed", "--seqs", str(fasta), "-o", emb, "-d", DEV])

    results = {}
    for regime in args.regimes:
        for arm, negs in arms.items():
            tag = f"{regime}_{arm}"
            if regime == "in_distribution":
                cut = int(len(pos) * 0.2)
                te_pos, tr_pos = pos[:cut], pos[cut:]
                tr_neg = list(negs)[:args.n]; te_neg = gold
            else:  # protein_disjoint
                tr_pos = [(u, v) for u, v in pos if side(u) == 0 and side(v) == 0]
                te_pos = [(u, v) for u, v in pos if side(u) == 1 and side(v) == 1]
                tr_neg = [(u, v) for u, v in negs if side(u) == 0 and side(v) == 0][:args.n]
                te_neg = [(u, v) for u, v in gold if side(u) == 1 and side(v) == 1]
            n = min(len(tr_pos), len(tr_neg), args.n)
            tr_pos = list(tr_pos)[:n]; tr_neg = list(tr_neg)[:n]
            te_pos = list(te_pos)[:max(200, len(te_neg))]

            tr_pairs = tr_pos + tr_neg
            tr_lab = [1] * len(tr_pos) + [0] * len(tr_neg)
            _write_pairs(WORK / f"{tag}_train.tsv", tr_pairs, tr_lab)
            te_pairs = te_pos + te_neg
            te_lab = [1] * len(te_pos) + [0] * len(te_neg)
            _write_pairs(WORK / f"{tag}_test.tsv", te_pairs, te_lab)

            print(f"\n== {tag}: train {len(tr_pos)}+/{len(tr_neg)}-  test {len(te_pos)}+/{len(te_neg)}-")
            prefix = str(WORK / f"{tag}_model")
            _run([DSCRIPT, "train", "--train", str(WORK / f"{tag}_train.tsv"),
                  "--test", str(WORK / f"{tag}_test.tsv"), "--embedding", emb,
                  "--num-epochs", str(args.epochs), "--save-prefix", prefix, "-d", DEV])
            model = f"{prefix}_final.sav"
            if not Path(model).exists():
                cand = sorted(WORK.glob(f"{tag}_model_epoch*.sav"))
                model = str(cand[-1]) if cand else model
            auroc, ntest = _predict_auroc(tag, te_pairs, te_lab, model, emb)
            results[(regime, arm)] = auroc
            print(f"   -> {tag} AUROC = {auroc:.3f}  (n_test={ntest})")

    print("\n" + "=" * 60)
    print("D-SCRIPT — AUROC by regime × arm (bounded run)")
    print("=" * 60)
    print(f"  {'regime':<20}{'random':>9}{'stacked':>9}{'verified':>9}")
    for regime in args.regimes:
        r = results.get((regime, "random"), float("nan"))
        s = results.get((regime, "stacked"), float("nan"))
        v = results.get((regime, "verified"), float("nan"))
        print(f"  {regime:<20}{r:>9.3f}{s:>9.3f}{v:>9.3f}   Δ(ver-rand) {v-r:+.3f}")
    WORK.joinpath("dscript_results.json").write_text(
        json.dumps({f"{r}|{a}": results[(r, a)] for (r, a) in results}, indent=2))
    print("\nwrote out/dscript/dscript_results.json")


if __name__ == "__main__":
    main()
