"""Fetch AlphaFold DB structures (mmCIF) + per-residue pLDDT confidence for
UniProt accessions, for use by scripts/compute_surface_hydrophobicity.py
(Tier 1: DSSP-derived exposure) and scripts/compute_pocket_descriptors.py
(fpocket).

Calls the AlphaFold DB prediction API (verified live:
https://alphafold.ebi.ac.uk/api/prediction/{accession} returns cifUrl +
plddtDocUrl + a mean-pLDDT globalMetricValue). A model is flagged
low_confidence when its mean pLDDT falls below MIN_PLDDT — callers should
treat that protein as "no usable structure" rather than trust the region.

Fetches run concurrently (MAX_WORKERS threads — checked live: EBI's terms of
use document no specific numeric rate limit for this API, just a general
"don't disrupt service for others" clause, so MAX_WORKERS is set
conservatively rather than maximized). A 429 (Too Many Requests) response
backs off with a longer, increasing delay than a generic error and is
retried on its own thread — it never aborts the whole batch.

    PYTHONPATH=. python scripts/fetch_alphafold_structures.py --ids-file uniprot_ids.txt

Writes (gitignored, like every other vendored dataset under local-docs/):
    local-docs/alphafold/<accession>.cif             the model
    local-docs/alphafold/<accession>.plddt.json      per-residue confidence
    local-docs/alphafold/manifest.tsv                accession<TAB>mean_plddt<TAB>low_confidence
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

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

OUT_DIR = Path("local-docs/alphafold")
MANIFEST = OUT_DIR / "manifest.tsv"
API = "https://alphafold.ebi.ac.uk/api/prediction"
MIN_PLDDT = 70.0          # AlphaFold's own "confident" band starts at 70
MAX_WORKERS = 6           # conservative — see module docstring


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL) as fh:
        return json.load(fh)


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh, open(path, "wb") as out:
        out.write(fh.read())


def fetch_one(accession: str, out_dir: Path = OUT_DIR) -> dict | None:
    """Fetch one accession's structure + confidence. Returns a manifest row
    dict, or None if AlphaFold has no model for this accession (not an error
    — many proteins genuinely have no prediction)."""
    try:
        entries = _get_json(f"{API}/{accession}")
    except Exception:
        return None
    if not entries:
        return None
    e = entries[0]
    cif_path = out_dir / f"{accession}.cif"
    plddt_path = out_dir / f"{accession}.plddt.json"
    if not cif_path.exists():
        _download(e["cifUrl"], cif_path)
    if not plddt_path.exists():
        _download(e["plddtDocUrl"], plddt_path)
    mean_plddt = float(e.get("globalMetricValue", 0.0))
    return {
        "accession": accession,
        "cif_path": str(cif_path),
        "plddt_path": str(plddt_path),
        "mean_plddt": mean_plddt,
        "low_confidence": mean_plddt < MIN_PLDDT,
    }


def _fetch_with_retry(accession: str, out_dir: Path) -> dict | None:
    row = None
    for attempt in range(4):
        try:
            row = fetch_one(accession, out_dir)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)          # longer backoff specifically for rate-limiting
                print(f"  {accession}: 429 rate-limited, backing off {wait}s")
                time.sleep(wait)
            else:
                print(f"  {accession}: HTTP {e.code}, retry {attempt + 1}")
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"  {accession}: retry {attempt + 1} ({e})")
            time.sleep(2 * (attempt + 1))
    return row


def fetch_many(accessions: list[str], out_dir: Path = OUT_DIR,
                max_workers: int = MAX_WORKERS) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_with_retry, acc, out_dir): acc for acc in accessions}
        done = 0
        for future in as_completed(futures):
            acc = futures[future]
            row = future.result()
            if row is not None:
                results[acc] = row
            done += 1
            if done % 50 == 0 or done == len(accessions):
                print(f"  fetched {done}/{len(accessions)}")
    return results


def _write_manifest(results: dict[str, dict], path: Path = MANIFEST) -> None:
    merged: dict[str, tuple[str, str]] = {}
    if path.exists():                                   # merge, don't clobber
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                parts = line.split("\t")
                merged[parts[0]] = (parts[1], parts[2])
    for acc, row in results.items():
        merged[acc] = (str(row["mean_plddt"]), str(row["low_confidence"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for acc in sorted(merged):
            plddt, low = merged[acc]
            fh.write(f"{acc}\t{plddt}\t{low}\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fetch_alphafold_structures")
    ap.add_argument("--ids-file", required=True)
    args = ap.parse_args(argv)
    ids = sorted(set(l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()))
    print(f"{len(ids)} UniProt accessions")

    results = fetch_many(ids)
    _write_manifest(results)
    n_conf = sum(1 for r in results.values() if not r["low_confidence"])
    print(f"fetched {len(results)}/{len(ids)} structures; "
          f"{n_conf} confident (mean pLDDT >= {MIN_PLDDT})")


if __name__ == "__main__":
    main()
