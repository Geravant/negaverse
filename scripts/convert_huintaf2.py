"""Turn the huintaf2 (Burke et al. 2023) AF2-Multimer CSVs into a signal column.

Reads local-docs/huintaf2/{HuRI,humap,random}.csv (each: Name=`id1-id2`, pDockQ,
IF_plDDT, …) — real AlphaFold2-Multimer runs on 65k+ human pairs — and:

  1. prints the real AF2 separation result (AUROC of pDockQ: interactions vs
     random pairs) — the honest "does a predicted-interface score tell real
     interactions apart" number, with no folding on our side;
  2. writes out/af2_scores.tsv (`id  id  1-pDockQ`) — higher = safer negative,
     the direction bench_rules --external expects.

HuRI.csv is Ensembl-keyed (matches our HuRI graph); humap/random are UniProt.

    PYTHONPATH=. python3 scripts/convert_huintaf2.py
    PYTHONPATH=. python3 scripts/bench_rules.py --dataset huintaf2 --external af2=out/af2_scores.tsv
"""
from __future__ import annotations

import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

_DIR = "local-docs/huintaf2"


def _load(name):
    """Return {(id_a, id_b): pDockQ} for pairs whose Name splits cleanly in two."""
    out, skipped = {}, 0
    with open(f"{_DIR}/{name}.csv") as fh:
        for r in csv.DictReader(fh):
            parts = r["Name"].split("-")
            if len(parts) != 2:
                skipped += 1
                continue
            try:
                out[(parts[0], parts[1])] = float(r["pDockQ"])
            except (KeyError, ValueError):
                skipped += 1
    return out, skipped


def _separation(pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    pv, nv = np.array(list(pos.values())), np.array(list(neg.values()))
    n = min(len(pv), len(nv))
    p = rng.choice(pv, n, replace=False)
    q = rng.choice(nv, n, replace=False)
    y = np.r_[np.ones(n), np.zeros(n)]
    return round(float(roc_auc_score(y, np.r_[p, q])), 3), n


def main():
    huri, s1 = _load("HuRI")
    humap, s2 = _load("humap")
    rand, s3 = _load("random")
    print(f"loaded: HuRI={len(huri)} humap={len(humap)} random={len(rand)} "
          f"(skipped {s1+s2+s3} malformed/isoform names)")
    print("\nReal AF2-Multimer pDockQ separation (no folding — published scores):")
    for nm, pos in [("HuRI", huri), ("Hu.MAP", humap)]:
        au, n = _separation(pos, rand)
        med = np.median(list(pos.values()))
        print(f"  {nm:7} vs random   AUROC(pDockQ) = {au:.3f}   "
              f"(median pDockQ {med:.3f}, n={n}/class)")
    med_r = np.median(list(rand.values()))
    print(f"  random  median pDockQ = {med_r:.3f}")

    os.makedirs("out", exist_ok=True)
    n_written = 0
    with open("out/af2_scores.tsv", "w") as fh:
        for table in (huri, humap, rand):
            for (a, b), q in table.items():
                fh.write(f"{a}\t{b}\t{1.0 - q:.4f}\n")   # safe-negative = 1 - pDockQ
                n_written += 1
    print(f"\nwrote out/af2_scores.tsv: {n_written} pairs (1 - pDockQ)")


if __name__ == "__main__":
    main()
