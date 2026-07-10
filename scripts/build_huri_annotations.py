"""Produce HuRI-space (Ensembl-gene) annotations so the biology rules fire on the
benchmark graph — the #1 gap for validating "independent signals beat topology".

HuRI nodes are Ensembl gene ids (ENSG); our rules read `compartments` (GO
cellular-component) and `surface_hydrophobicity`. This maps HuRI's ENSG nodes to
UniProt via the human reviewed proteome, pulls GO CC + sequence, and MERGES:
  * ENSG -> compartments   into local-docs/localization/go_cc.tsv
  * ENSG -> hydrophobicity into local-docs/annotations/hydrophobicity.tsv
so build_annotation_table() picks them up automatically (co-localization is then an
independent, non-topology signal on HuRI).

    PYTHONPATH=. python scripts/build_huri_annotations.py

Network required (UniProt REST, paginated). ENSG keys never collide with the
UniProt keys already in those files, so SARS annotations are preserved.
"""
from __future__ import annotations

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

LOC = Path("local-docs/localization/go_cc.tsv")
HYD = Path("local-docs/annotations/hydrophobicity.tsv")
SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ACC = "https://rest.uniprot.org/uniprotkb/accessions"

_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
       "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
       "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        link = fh.headers.get("Link", "")
        data = json.load(fh)
    # the next-page URL contains commas (fields=a,b,c), so match the whole <...>
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return data, (m.group(1) if m else None)


def _iter_proteome():
    url = SEARCH + "?" + urllib.parse.urlencode({
        "query": "organism_id:9606 AND reviewed:true",
        "fields": "accession,xref_ensembl,go_c", "format": "json", "size": "500"})
    page = 0
    while url:
        for attempt in range(3):
            try:
                data, url = _get(url)
                break
            except Exception as e:
                print(f"  page {page}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))
        else:
            break
        for e in data.get("results", []):
            yield e
        page += 1
        if page % 10 == 0:
            print(f"  proteome page {page} ...")


def _score(seq):
    vals = [_KD[a] for a in seq if a in _KD]
    if not vals:
        return None
    return round(max(0.0, min(1.0, (sum(vals) / len(vals) + 4.5) / 9.0)), 4)


def _fetch_seqs(accs):
    q = urllib.parse.urlencode({"accessions": ",".join(accs),
                                "fields": "accession,sequence", "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(f"{ACC}?{q}"), timeout=30, context=_SSL) as fh:
        data = json.load(fh)
    return {e["primaryAccession"]: e.get("sequence", {}).get("value")
            for e in data.get("results", []) if e.get("sequence")}


def _merge(path: Path, new: dict, joiner):
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                cur[k] = v
    for k, v in new.items():
        cur[k] = joiner(v)
    path.write_text("".join(f"{k}\t{cur[k]}\n" for k in sorted(cur)))
    return len(new), len(cur)


def main():
    from negaverse.io import load_huri_graph
    huri = set(load_huri_graph().g.nodes())
    print(f"HuRI ENSG nodes: {len(huri):,}")

    ensg_comp: dict[str, set] = {}
    ensg_uni: dict[str, str] = {}
    print("Scanning human reviewed proteome (UniProt) ...")
    for e in _iter_proteome():
        acc = e.get("primaryAccession")
        ensgs, comps = set(), set()
        for x in e.get("uniProtKBCrossReferences", []):
            db = x.get("database")
            if db == "Ensembl":
                for p in x.get("properties", []):
                    if p.get("key") == "GeneId" and p.get("value"):
                        ensgs.add(p["value"].split(".")[0])
            elif db == "GO":
                for p in x.get("properties", []):
                    if p.get("key") == "GoTerm" and p.get("value", "").startswith("C:"):
                        comps.add(p["value"][2:].strip().lower())
        for g in ensgs & huri:
            if comps:
                ensg_comp.setdefault(g, set()).update(comps)
            ensg_uni.setdefault(g, acc)
    print(f"  mapped {len(ensg_uni):,} HuRI genes to UniProt; "
          f"{len(ensg_comp):,} have GO compartments")

    # hydrophobicity for the mapped UniProts
    print("Fetching sequences + scoring hydrophobicity ...")
    uni_list = sorted(set(ensg_uni.values()))
    seqs: dict[str, str] = {}
    for i in range(0, len(uni_list), 100):
        for attempt in range(3):
            try:
                seqs.update(_fetch_seqs(uni_list[i:i + 100]))
                break
            except Exception as ex:
                print(f"  seq chunk {i//100}: retry {attempt+1} ({ex})")
                time.sleep(2 * (attempt + 1))
    ensg_hyd = {}
    for g, acc in ensg_uni.items():
        s = seqs.get(acc)
        if s and (sc := _score(s)) is not None:
            ensg_hyd[g] = sc

    n1, t1 = _merge(LOC, ensg_comp, lambda s: ",".join(sorted(s)))
    n2, t2 = _merge(HYD, ensg_hyd, lambda v: str(v))
    print(f"\nmerged {n1} HuRI compartment rows into {LOC} (total {t1})")
    print(f"merged {n2} HuRI hydrophobicity rows into {HYD} (total {t2})")


if __name__ == "__main__":
    main()
