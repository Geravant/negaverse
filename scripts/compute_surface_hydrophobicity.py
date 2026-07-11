"""Two-tier per-protein surface-hydrophobicity: structure-based (Tier 1) when
a confident AlphaFold model exists, sequence-based fallback (Tier 2 — the
original scripts/compute_hydrophobicity.py proxy) otherwise. Supersedes that
script's output for the `surface_hydrophobicity` annotation field; kept
alongside it since compute_hydrophobicity.py is still the simplest path for a
quick, structure-free estimate.

Tier 1 (mean pLDDT >= fetch_alphafold_structures.MIN_PLDDT, i.e. a usable
AlphaFold model exists): run mkdssp (via Bio.PDB.DSSP, which handles the
classic DSSP format's edge cases — disulfide cysteines shown lowercase, chain
breaks, etc. — far more robustly than a hand-rolled column parser) on the
AlphaFold mmCIF model to get each residue's relative solvent accessibility
(RSA, Bio.PDB.DSSP normalizes this internally). Aggregate Kyte-Doolittle over
residues that are both:
  - solvent-exposed (RSA >= RSA_CUTOFF), and
  - ordered/confident (per-residue pLDDT >= PLDDT_CUTOFF, from the same
    manifest fetch_alphafold_structures.py already downloaded) — this is the
    disorder-masking step, using AlphaFold's own confidence as the disorder
    proxy instead of a separate disorder predictor.

Tier 2 (no usable structure): mean Kyte-Doolittle over the whole sequence,
identical to scripts/compute_hydrophobicity.py.

DSSP results are cached per structure (local-docs/alphafold/<acc>.dssp.json)
since re-running mkdssp is the dominant cost of this pipeline — see
_get_dssp_residues.

Requires `mkdssp` on PATH (conda-forge: `mamba install -c conda-forge dssp`)
and biopython (`pip install biopython`). If `mkdssp`'s bundled Chemical
Component Dictionary isn't found via LIBCIFPP_DATA_DIR, this script points it
at the conda-forge install layout automatically; override the env var
yourself if you installed dssp differently.

    PYTHONPATH=. python scripts/compute_surface_hydrophobicity.py --dataset sars
    PYTHONPATH=. python scripts/compute_surface_hydrophobicity.py --ids-file uniprot_ids.txt

Writes (merge, not clobber — same convention as compute_hydrophobicity.py):
    local-docs/annotations/hydrophobicity.tsv          node<TAB>score
    local-docs/annotations/hydrophobicity_tier.tsv     node<TAB>{structure,sequence}

Note on direction: calibration against DRYAD + UPNA-PPI (both real gold-standard
PPI benchmarks, scripts/calibrate_hydrophobicity_threshold.py) showed this score
is *inversely* related to interaction likelihood — real interactions tend to
have LOWER exposed hydrophobicity than real non-interactions, consistent with
PPI interface hot-spots being enriched in aromatic/cation-pi residues (Trp,
Tyr, Arg — Bogan & Thorn 1998) rather than classic Kyte-Doolittle-hydrophobic
ones. rules/ppi.yaml's hydrophobicity_interface rule fires on *high* (not low)
values accordingly — see that rule's rationale/source for the calibrated
threshold and citations.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.compute_hydrophobicity import _KD, _sars_ids
from scripts.fetch_alphafold_structures import fetch_many

# point mkdssp's Chemical Component Dictionary at the conda-forge install
# layout if not already configured elsewhere
_CONDA_CIFPP = "/opt/homebrew/Caskroom/miniforge/base/share/libcifpp"
if "LIBCIFPP_DATA_DIR" not in os.environ and Path(_CONDA_CIFPP).is_dir():
    os.environ["LIBCIFPP_DATA_DIR"] = _CONDA_CIFPP

OUT = Path("local-docs/annotations/hydrophobicity.tsv")
TIER_OUT = Path("local-docs/annotations/hydrophobicity_tier.tsv")

RSA_CUTOFF = 0.25     # solvent-exposed threshold; 25% RSA is a common "exposed" convention
PLDDT_CUTOFF = 70.0   # disorder-proxy cutoff, matches AlphaFold's own "confident" band

_KD_RANGE = (-4.5, 4.5)


def _kd_score(vals: list[float]) -> float:
    mean = sum(vals) / len(vals)
    lo, hi = _KD_RANGE
    return round(max(0.0, min(1.0, (mean - lo) / (hi - lo))), 4)


def _dssp_cache_path(cif_path: Path) -> Path:
    return cif_path.with_suffix("").with_suffix(".dssp.json")


def _get_dssp_residues(accession: str, cif_path: Path) -> list[dict] | None:
    """Run DSSP once per structure and cache the result (RSA + amino acid
    identity per residue) — this is the dominant cost of the pipeline, so
    never re-run it for a structure already scored once. Cache is a JSON list
    of {resnum, aa, rsa} per residue DSSP could assign an AA + RSA to (no
    exposure/pLDDT filtering yet — that's applied by the caller, so changing
    RSA_CUTOFF/PLDDT_CUTOFF doesn't require re-running DSSP either). Returns
    None if DSSP fails to run on this structure at all.
    """
    cache_path = _dssp_cache_path(cif_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    from Bio.PDB import MMCIFParser
    from Bio.PDB.DSSP import DSSP

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(accession, str(cif_path))
    try:
        dssp = DSSP(structure[0], str(cif_path), dssp="mkdssp", file_type="MMCIF")
    except Exception:
        return None

    residues = []
    for key in dssp.keys():
        _, (_, resnum, _) = key
        row = dssp[key]
        aa, rel_asa = row[1], row[3]
        if not isinstance(rel_asa, (int, float)):
            continue
        residues.append({"resnum": int(resnum), "aa": aa.upper(), "rsa": float(rel_asa)})
    cache_path.write_text(json.dumps(residues))
    return residues


def _structure_score(accession: str, cif_path: Path, plddt_path: Path) -> float | None:
    """Tier 1. None if DSSP fails to run or no residue qualifies (caller
    should fall back to Tier 2 in that case, not fabricate a value)."""
    plddt_by_resnum: dict[int, float] = {}
    data = json.loads(plddt_path.read_text())
    for resnum, conf in zip(data["residueNumber"], data["confidenceScore"]):
        plddt_by_resnum[int(resnum)] = float(conf)

    residues = _get_dssp_residues(accession, cif_path)
    if residues is None:
        return None

    vals = []
    for res in residues:
        aa, rel_asa, resnum = res["aa"], res["rsa"], res["resnum"]
        if aa not in _KD:
            continue
        plddt = plddt_by_resnum.get(resnum)
        if plddt is None or plddt < PLDDT_CUTOFF or rel_asa < RSA_CUTOFF:
            continue
        vals.append(_KD[aa])
    return _kd_score(vals) if vals else None


def _fetch_sequences(ids: list[str]) -> dict[str, float]:
    """Tier-2 baseline for every accession, batched/retried like
    compute_hydrophobicity.main()."""
    import ssl
    import urllib.parse
    import urllib.request
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_ctx = ssl._create_unverified_context()

    scores: dict[str, float] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for attempt in range(3):
            try:
                q = urllib.parse.urlencode({"accessions": ",".join(chunk),
                                            "fields": "accession,sequence", "format": "json"})
                req = urllib.request.Request(
                    f"https://rest.uniprot.org/uniprotkb/accessions?{q}",
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as fh:
                    data = json.load(fh)
                for e in data.get("results", []):
                    acc = e.get("primaryAccession")
                    seq = e.get("sequence", {}).get("value")
                    if acc and seq:
                        vals = [_KD[a] for a in seq if a in _KD]
                        if vals:
                            scores[acc] = _kd_score(vals)
                break
            except Exception as e:
                print(f"  sequence chunk {i // 100}: retry {attempt + 1} ({e})")
                time.sleep(2 * (attempt + 1))
    return scores


def compute(accessions: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    """Returns (scores, tiers); tiers[acc] in {"structure", "sequence"}."""
    print(f"Tier 2 (sequence) baseline for {len(accessions)} accessions...")
    scores = _fetch_sequences(accessions)
    tiers = {a: "sequence" for a in scores}

    print("fetching AlphaFold structures for Tier 1 (structure) upgrade...")
    af = fetch_many(accessions)
    candidates = [(acc, row) for acc, row in af.items() if not row["low_confidence"]]

    # DSSP is a local, CPU-bound external subprocess (mkdssp) — unlike the
    # AlphaFold API fetch, there's no server to be polite to, so parallelize
    # up to the machine's core count rather than a conservative constant.
    n_workers = min(len(candidates), os.cpu_count() or 4) or 1
    n_upgraded = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_structure_score, acc, Path(row["cif_path"]), Path(row["plddt_path"])): acc
            for acc, row in candidates
        }
        for future in as_completed(futures):
            acc = futures[future]
            s = future.result()
            if s is not None:
                scores[acc] = s
                tiers[acc] = "structure"
                n_upgraded += 1
    print(f"{n_upgraded}/{len(accessions)} upgraded to Tier 1 (structure-based)")
    return scores, tiers


def _write_merged(path: Path, new_values: dict[str, str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                merged[k] = v
    merged.update(new_values)
    with open(path, "w") as fh:
        for n in sorted(merged):
            fh.write(f"{n}\t{merged[n]}\n")
    return len(merged)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_surface_hydrophobicity")
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

    scores, tiers = compute(ids)
    n_scores = _write_merged(OUT, {k: str(v) for k, v in scores.items()})
    n_tiers = _write_merged(TIER_OUT, tiers)
    print(f"wrote {len(scores)} new / {n_scores} total scores to {OUT}")
    print(f"wrote {len(tiers)} new / {n_tiers} total tier labels to {TIER_OUT}")


if __name__ == "__main__":
    main()
