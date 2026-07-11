"""Compute a per-protein surface-hydrophobicity proxy and write the annotation
TSV the hydrophobicity rule reads.

Fetches sequences from UniProt and scores each protein by its normalized mean
Kyte-Doolittle hydrophobicity in [0,1] (0 = very hydrophilic, 1 = very
hydrophobic). This is a **sequence-level proxy** — a proper surface/interface
hydrophobicity needs structure. Superseded for the `surface_hydrophobicity`
annotation field by scripts/compute_surface_hydrophobicity.py, which reuses
this module's UniProt fetch + Kyte-Doolittle scale as its own Tier 2 (no
usable structure) fallback and adds a structure-aware Tier 1 on top; run that
script instead unless you specifically want the plain sequence-only estimate
with no structure lookup at all.

    PYTHONPATH=. python scripts/compute_hydrophobicity.py --dataset sars
    PYTHONPATH=. python scripts/compute_hydrophobicity.py --ids-file uniprot_ids.txt

Writes local-docs/annotations/hydrophobicity.tsv  (node<TAB>score), gitignored.
Then build_annotation_table() picks it up automatically.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

OUT = Path("local-docs/annotations/hydrophobicity.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/accessions"
_UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
       "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
       "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def _score(seq: str) -> float | None:
    vals = [_KD[a] for a in seq if a in _KD]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return round(max(0.0, min(1.0, (mean + 4.5) / 9.0)), 4)   # normalize KD [-4.5,4.5] -> [0,1]


def _fetch(accessions: list[str]) -> dict[str, float]:
    q = urllib.parse.urlencode({"accessions": ",".join(accessions),
                                "fields": "accession,sequence", "format": "json"})
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL) as fh:
        data = json.load(fh)
    out: dict[str, float] = {}
    for e in data.get("results", []):
        acc = e.get("primaryAccession")
        seq = e.get("sequence", {}).get("value")
        if acc and seq:
            s = _score(seq)
            if s is not None:
                out[acc] = s
    return out


def _sars_ids() -> list[str]:
    from negaverse.io import load_sars_cov2_graph
    g = load_sars_cov2_graph()
    return [n for n in g.g.nodes() if _UNIPROT_RE.match(n)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_hydrophobicity")
    ap.add_argument("--dataset", choices=["sars"])
    ap.add_argument("--ids-file")
    args = ap.parse_args(argv)
    if args.ids_file:
        ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    elif args.dataset == "sars":
        ids = _sars_ids()
    else:
        ap.error("give --dataset or --ids-file")
    ids = sorted(set(ids))
    print(f"{len(ids)} UniProt accessions")

    scores: dict[str, float] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for attempt in range(3):
            try:
                scores.update(_fetch(chunk))
                break
            except Exception as e:
                print(f"  chunk {i//100}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if OUT.exists():                                    # merge, don't clobber other datasets
        for line in OUT.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                merged[k] = v
    for n, s in scores.items():
        merged[n] = str(s)
    with open(OUT, "w") as fh:
        for n in sorted(merged):
            fh.write(f"{n}\t{merged[n]}\n")
    print(f"wrote {len(scores)} new / {len(merged)} total scores to {OUT}")


if __name__ == "__main__":
    main()
