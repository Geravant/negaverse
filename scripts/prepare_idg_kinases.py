"""Prepare the IDG understudied-kinome candidate space (open data).

Pulls the human understudied kinases (Target Development Level Tdark + Tbio,
Family=Kinase) from Pharos/IDG, fetches their sequences from UniProt, and writes:

    local-docs/idg/kinases.tsv     uniprot <tab> symbol <tab> tdl <tab> length
    local-docs/idg/sequences.tsv   uniprot <tab> sequence     (ESM2 build input)

Then build embeddings with:
    PYTHONPATH=. python3 scripts/build_esm2_embeddings.py \
        --seqs local-docs/idg/sequences.tsv --out local-docs/idg/esm2_idg.npz

Sources (all open): Pharos GraphQL API (pharos-api.ncats.io, IDG/NIH, CC-BY),
UniProtKB REST (rest.uniprot.org, CC-BY 4.0). Nothing here is hard-coded biology;
re-run to refresh as IDG re-classifies targets.

    PYTHONPATH=. python3 scripts/prepare_idg_kinases.py
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

PHAROS = "https://pharos-api.ncats.io/graphql"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
OUT = Path("local-docs/idg")
_SSL = ssl._create_unverified_context()   # macOS python often lacks CA certs (cf. viz/interactive.py)


def _post_json(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return json.loads(r.read().decode())


def pharos_kinases(tdls=("Tdark", "Tbio")) -> list[dict]:
    """Understudied kinases = targets with these TDLs and Family == Kinase.
    (Pharos's Family facet doesn't filter server-side, so we page the working
    TDL facet and filter fam client-side.)"""
    out: dict[str, dict] = {}
    for tdl in tdls:
        q = ('{ targets(filter:{facets:[{facet:"Target Development Level",'
             'values:["%s"]}]}) { count targets(top:15000){ uniprot sym fam tdl } } }' % tdl)
        d = _post_json(PHAROS, {"query": q})
        t = d["data"]["targets"]
        got = t["targets"]
        if len(got) < t["count"]:
            print(f"  WARNING {tdl}: got {len(got)}/{t['count']} (raise top:) — some may be dropped")
        for x in got:
            if x["fam"] == "Kinase" and x.get("uniprot"):
                out[x["uniprot"]] = {"uniprot": x["uniprot"], "sym": x["sym"], "tdl": x["tdl"]}
        print(f"  {tdl}: {sum(1 for x in got if x['fam']=='Kinase')} kinases")
    return sorted(out.values(), key=lambda r: (r["tdl"], r["sym"]))


def fetch_sequences(accs: list[str], chunk: int = 100) -> dict[str, str]:
    """UniProt canonical sequences for a list of accessions, chunked FASTA."""
    seqs: dict[str, str] = {}
    for i in range(0, len(accs), chunk):
        part = accs[i:i + chunk]
        query = " OR ".join(f"accession:{a}" for a in part)
        url = UNIPROT + "?" + urllib.parse.urlencode(
            {"query": query, "format": "fasta", "size": str(chunk)})
        with urllib.request.urlopen(url, timeout=120, context=_SSL) as r:
            fasta = r.read().decode()
        acc, buf = None, []
        for line in fasta.splitlines():
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc = line.split("|")[1] if "|" in line else None
                buf = []
            else:
                buf.append(line.strip())
        if acc:
            seqs[acc] = "".join(buf)
        print(f"  fetched {min(i+chunk, len(accs))}/{len(accs)}")
        time.sleep(0.3)
    return seqs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Querying Pharos for understudied kinases (Tdark + Tbio) ...")
    kinases = pharos_kinases()
    print(f"  {len(kinases)} understudied kinases")

    print("Fetching sequences from UniProt ...")
    seqs = fetch_sequences([k["uniprot"] for k in kinases])
    missing = [k["uniprot"] for k in kinases if k["uniprot"] not in seqs]
    if missing:
        print(f"  {len(missing)} without a sequence (dropped): {missing[:10]}")

    kinases = [k for k in kinases if k["uniprot"] in seqs]
    with open(OUT / "kinases.tsv", "w") as fh:
        fh.write("uniprot\tsymbol\ttdl\tlength\n")
        for k in kinases:
            fh.write(f"{k['uniprot']}\t{k['sym']}\t{k['tdl']}\t{len(seqs[k['uniprot']])}\n")
    with open(OUT / "sequences.tsv", "w") as fh:
        for k in kinases:
            fh.write(f"{k['uniprot']}\t{seqs[k['uniprot']]}\n")

    from collections import Counter
    print(f"\nwrote {len(kinases)} kinases -> {OUT}/kinases.tsv, sequences.tsv")
    print("  by TDL:", dict(Counter(k["tdl"] for k in kinases)))
    print(f"\nnext: PYTHONPATH=. python3 scripts/build_esm2_embeddings.py "
          f"--seqs {OUT}/sequences.tsv --out {OUT}/esm2_idg.npz")


if __name__ == "__main__":
    main()
