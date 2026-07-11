"""Fetch orthologous CDS sequences for UniProt accessions, for use by
scripts/build_gene_alignments.py (MAFFT) and ultimately
scripts/compute_evolutionary_coupling.py (Evolutionary Rate Covariation via
RERconverge).

UniProt -> Ensembl gene (ENSG) mapping reuses
scripts/build_uniprot_ensembl_map.py::_fetch (already the UniProt->ENSG mapper
in this repo, previously hardcoded to Negatome IDs — the fetch logic itself is
generic).

Calls Ensembl's REST homology endpoint (verified live:
https://rest.ensembl.org/homology/id/{species}/{ensembl_id}?type=orthologues
returns every ortholog Ensembl has for that gene in ONE call — no need for a
separate call per target species). Filters the response to a fixed
SPECIES_PANEL (see below) so every gene's tree is built over a comparable set
of species; RERconverge's rate covariation depends on genes sharing (mostly)
the same species panel, not on using every species Ensembl happens to have.

Ensembl's homology response gives `align_seq` per ortholog — a *pairwise*
alignment against the query gene (gapped only against that one pair), not a
raw sequence. Stripping '-' recovers the raw (unaligned) CDS, which is what we
want here since scripts/build_gene_alignments.py builds its own multi-species
MSA across the whole panel with MAFFT.

SPECIES_PANEL is a fixed, broad vertebrate panel (primates -> rodents ->
carnivores -> ungulates -> marsupials -> monotremes -> birds -> amphibians ->
fish) chosen for phylogenetic spread and Ensembl annotation quality — a
starting default, easy to widen/narrow later, not a hard commitment.

    PYTHONPATH=. python scripts/fetch_orthologs.py --ids-file uniprot_ids.txt

Writes (gitignored, one file per gene, merge/skip-if-cached like
local-docs/alphafold/*.cif):
    local-docs/orthologs/<ensembl_gene_id>.json
        {"species": {"ensembl_gene_id", "taxon_id", "homology_type", "cds"}, ...}
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.build_uniprot_ensembl_map import _fetch as _fetch_uniprot_to_ensembl

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

OUT_DIR = Path("local-docs/orthologs")
API = "https://rest.ensembl.org/homology/id/human"
MAX_WORKERS = 6          # conservative, matches fetch_alphafold_structures.py's convention

# Fixed vertebrate species panel (Ensembl production names). Easy to widen or
# narrow later — this is a starting default, not a hard commitment.
SPECIES_PANEL = [
    "pan_troglodytes", "macaca_mulatta", "mus_musculus", "rattus_norvegicus",
    "canis_lupus_familiaris", "felis_catus", "bos_taurus", "sus_scrofa",
    "equus_caballus", "oryctolagus_cuniculus", "monodelphis_domestica",
    "ornithorhynchus_anatinus", "gallus_gallus", "xenopus_tropicalis",
    "danio_rerio",
]


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL) as fh:
        return json.load(fh)


def _fetch_with_retry(url: str) -> dict | None:
    for attempt in range(4):
        try:
            return _get_json(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                       # gene not found / no homology data
            wait = 5 * (attempt + 1) if e.code == 429 else 2 * (attempt + 1)
            time.sleep(wait)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_one_gene(ensembl_id: str, out_dir: Path = OUT_DIR) -> dict | None:
    """Fetch + cache one gene's orthologs restricted to SPECIES_PANEL. Returns
    {species: {...}} or None if Ensembl has no homology data for this gene."""
    cache_path = out_dir / f"{ensembl_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    data = _fetch_with_retry(f"{API}/{ensembl_id}?type=orthologues;sequence=cds")
    if not data or not data.get("data"):
        return None
    homologies = data["data"][0].get("homologies", [])
    if not homologies:
        return None
    panel = set(SPECIES_PANEL)
    by_species: dict[str, dict] = {}

    # the query (human) sequence is only ever in each homology's `source` block,
    # identical across every entry — grab it once from the first, so the
    # resulting alignment/tree includes human alongside its orthologs.
    source = homologies[0].get("source", {})
    if source.get("align_seq"):
        by_species["homo_sapiens"] = {
            "ensembl_gene_id": source.get("id"),
            "taxon_id": source.get("taxon_id"),
            "homology_type": "self",
            "cds": source["align_seq"].replace("-", ""),
        }

    for h in homologies:
        target = h.get("target", {})
        species = target.get("species")
        if species not in panel or species in by_species:
            continue                              # keep first (Ensembl's own ranking)
        align_seq = target.get("align_seq")
        if not align_seq:
            continue
        by_species[species] = {
            "ensembl_gene_id": target.get("id"),
            "taxon_id": target.get("taxon_id"),
            "homology_type": h.get("type"),
            "cds": align_seq.replace("-", ""),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(by_species))
    return by_species


def fetch_many_genes(ensembl_ids: list[str], out_dir: Path = OUT_DIR,
                      max_workers: int = MAX_WORKERS) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one_gene, gid, out_dir): gid for gid in ensembl_ids}
        done = 0
        for future in as_completed(futures):
            gid = futures[future]
            row = future.result()
            if row:
                results[gid] = row
            done += 1
            if done % 25 == 0 or done == len(ensembl_ids):
                print(f"  fetched {done}/{len(ensembl_ids)}")
    return results


def map_accessions_to_genes(uniprot_accessions: list[str]) -> dict[str, str]:
    """UniProt accession -> one Ensembl gene ID (first mapped gene — a UniProt
    entry can cross-reference more than one Ensembl gene ID in rare cases;
    take one consistently rather than fetching/aligning against several).
    Accessions with no Ensembl mapping are simply absent."""
    print(f"mapping {len(uniprot_accessions)} UniProt accessions to Ensembl genes...")
    uniprot_to_ensg: dict[str, set[str]] = {}
    for i in range(0, len(uniprot_accessions), 100):
        chunk = uniprot_accessions[i:i + 100]
        for attempt in range(3):
            try:
                uniprot_to_ensg.update(_fetch_uniprot_to_ensembl(chunk))
                break
            except Exception as e:
                print(f"  chunk {i // 100}: retry {attempt + 1} ({e})")
                time.sleep(2 * (attempt + 1))
    acc_to_gene = {acc: sorted(genes)[0] for acc, genes in uniprot_to_ensg.items() if genes}
    print(f"  mapped {len(acc_to_gene)}/{len(uniprot_accessions)} accessions to an Ensembl gene")
    return acc_to_gene


def fetch_many(uniprot_accessions: list[str]) -> dict[str, dict]:
    """UniProt accession -> {species: {...}} ortholog data. Accessions with no
    Ensembl gene mapping, or no Ensembl homology data, are simply absent (no
    fabrication) — same convention as fetch_alphafold_structures.fetch_many."""
    acc_to_gene = map_accessions_to_genes(uniprot_accessions)
    gene_data = fetch_many_genes(sorted(set(acc_to_gene.values())))
    return {acc: gene_data[gene] for acc, gene in acc_to_gene.items() if gene in gene_data}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fetch_orthologs")
    ap.add_argument("--ids-file", required=True, help="UniProt accessions, one per line")
    args = ap.parse_args(argv)
    ids = sorted(set(l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()))
    print(f"{len(ids)} UniProt accessions")

    results = fetch_many(ids)
    n_species = [len(v) for v in results.values()]     # includes human, so max is panel+1
    avg = sum(n_species) / len(n_species) if n_species else 0
    print(f"fetched ortholog data for {len(results)}/{len(ids)} accessions "
          f"(avg {avg:.1f}/{len(SPECIES_PANEL) + 1} species covered, incl. human)")


if __name__ == "__main__":
    main()
