"""Real AF2-Multimer interface scores (pDockQ) -> the external column bench_rules eats.

AF2-Multimer needs a deep MSA + GPU inference, so it can't run in a plain sandbox
— this is the actual pipeline researchers use (ColabFold), wired end-to-end:

  pairs → fetch UniProt sequences → ColabFold (MMseqs2 MSA + AF2-Multimer) →
  pDockQ from the top model → out/af2_scores.tsv  (uniprot_a  uniprot_b  pDockQ)

then:  PYTHONPATH=. python3 scripts/bench_rules.py --dataset dryad \
           --external af2=out/af2_scores.tsv

pDockQ is the Bryant et al. 2022 interface-confidence score (0–1); high = a
confident predicted interface = likely real interaction => a *risky* negative.

Stages (each can run independently so the GPU step is isolated):
  --stage seqs      fetch sequences + write the ColabFold FASTA         (no GPU)
  --stage fold      run ColabFold on the FASTA                          (GPU)
  --stage score     compute pDockQ from the predicted PDBs -> TSV       (no GPU)

    PYTHONPATH=. python3 scripts/compute_af2_multimer.py --stage seqs --dataset dryad --n 20
    colabfold_batch out/af2/input.fasta out/af2/preds          # on a GPU box
    PYTHONPATH=. python3 scripts/compute_af2_multimer.py --stage score
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import time
import urllib.request

import numpy as np

_OUT = "out/af2"
_FASTA = f"{_OUT}/input.fasta"
_PREDS = f"{_OUT}/preds"
_TSV = "out/af2_scores.tsv"


# ---- inputs ------------------------------------------------------------
def _dryad_pairs(n, seed):
    import random
    rng = random.Random(seed)
    pos, neg = [], []
    path = "local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv"
    with open(path) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:n // 2] + neg[:n // 2]                # a mix so the column spans both classes


def _fetch_sequence(acc: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            body = r.read().decode()
        return "".join(body.splitlines()[1:]) or None   # drop the FASTA header
    except Exception:
        return None


def stage_seqs(pairs):
    os.makedirs(_OUT, exist_ok=True)
    accs = sorted({a for p in pairs for a in p})
    seqs: dict[str, str] = {}
    for i, acc in enumerate(accs):
        s = _fetch_sequence(acc)
        if s:
            seqs[acc] = s
        print(f"  fetched {i+1}/{len(accs)} ({len(seqs)} ok)", end="\r")
        time.sleep(0.1)                                  # be polite to UniProt
    print()
    with open(_FASTA, "w") as fh:
        for a, b in pairs:
            if a in seqs and b in seqs:
                fh.write(f">{a}__{b}\n{seqs[a]}:{seqs[b]}\n")   # ColabFold multimer format
    n = sum(1 for a, b in pairs if a in seqs and b in seqs)
    print(f"wrote {_FASTA}: {n}/{len(pairs)} pairs (both sequences available)")


def stage_fold():
    if not _which("colabfold_batch"):
        print("colabfold_batch not found. Install (needs a GPU):")
        print("  pip install 'colabfold[alphafold]'   # or use the ColabFold Docker/Colab")
        print(f"then: colabfold_batch {_FASTA} {_PREDS}")
        return
    os.makedirs(_PREDS, exist_ok=True)
    subprocess.run(["colabfold_batch", _FASTA, _PREDS], check=True)


def _which(x):
    from shutil import which
    return which(x)


# ---- pDockQ (Bryant et al. 2022) ---------------------------------------
def _parse_pdb(path):
    """Return per-chain lists of (resid, Cβ-or-Cα xyz, plDDT=B-factor)."""
    chains: dict[str, dict[int, tuple]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom, resn, chain = line[12:16].strip(), line[17:20].strip(), line[21]
            resid = int(line[22:26])
            want = "CB" if resn != "GLY" else "CA"
            if atom != want:
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            plddt = float(line[60:66])
            chains.setdefault(chain, {})[resid] = (xyz, plddt)
    return chains


def _pdockq(path, contact_thresh=8.0):
    chains = _parse_pdb(path)
    if len(chains) < 2:
        return None
    (ca, ra), (cb, rb) = list(chains.items())[:2]
    A = np.array([v[0] for v in ra.values()]); pA = np.array([v[1] for v in ra.values()])
    B = np.array([v[0] for v in rb.values()]); pB = np.array([v[1] for v in rb.values()])
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    ia, ib = np.where(d < contact_thresh)
    if len(ia) == 0:
        return 0.0
    n_contacts = len(ia)
    if_plddt = np.concatenate([pA[np.unique(ia)], pB[np.unique(ib)]]).mean()
    x = if_plddt * math.log(n_contacts)
    return round(0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018, 4)


def stage_score():
    rows = []
    for pdb in sorted(glob.glob(f"{_PREDS}/*_relaxed_rank_001*.pdb")
                      or glob.glob(f"{_PREDS}/*rank_001*.pdb")):
        base = os.path.basename(pdb)
        ident = base.split("_rank")[0].split("_unrelaxed")[0].split("_relaxed")[0]
        if "__" not in ident:
            continue
        a, b = ident.split("__")[:2]
        s = _pdockq(pdb)
        if s is not None:
            rows.append((a, b, s))
    with open(_TSV, "w") as fh:
        for a, b, s in rows:
            fh.write(f"{a}\t{b}\t{s}\n")
    print(f"wrote {_TSV}: {len(rows)} pDockQ scores")
    if not rows:
        print(f"  (no predictions found under {_PREDS} — run --stage fold first)")


def main():
    ap = argparse.ArgumentParser(prog="compute_af2_multimer")
    ap.add_argument("--stage", choices=["seqs", "fold", "score"], required=True)
    ap.add_argument("--dataset", choices=["dryad"], default="dryad")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "seqs":
        stage_seqs(_dryad_pairs(args.n, args.seed))
    elif args.stage == "fold":
        stage_fold()
    else:
        stage_score()


if __name__ == "__main__":
    main()
