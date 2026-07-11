"""Compute structure-based pocket descriptors (pocket_volume, pocket_hydrophobicity,
pocket_polarity) for proteins with a confident AlphaFold structure, using fpocket.

For each accession with a usable AlphaFold model (mean pLDDT >=
fetch_alphafold_structures.MIN_PLDDT), runs `fpocket` on the mmCIF file and reads
the **top-ranked pocket** (fpocket's own "Pocket 1", ordered by its internal
druggability-adjacent Score — verified live: Score strictly decreases from
Pocket 1 onward in fpocket's output) from its info.txt:
    pocket_volume          "Volume" (Angstrom^3), used as-is.
    pocket_hydrophobicity  "Hydrophobicity score", fpocket's native
                           (unnormalized) scale — no existing rule reads this
                           field yet, so no forced [0,1] normalization here;
                           document its native scale honestly instead of
                           fabricating false precision.
    pocket_polarity        "Proportion of polar atoms" (a percentage in
                           fpocket's output) divided by 100 -> a [0,1]
                           fraction, matching AUTHORING.md's documented
                           meaning ("fraction of polar residues/atoms") and
                           rules/pli.yaml's physicochemical_incompatibility
                           rule, which compares it numerically.

No structure (missing or low-confidence) -> all three fields simply absent
for that node, no fabrication — matching AUTHORING.md's existing "not all
proteins will have structures; rules must tolerate missing pocket fields"
guidance.

Requires `fpocket` on PATH (conda-forge: `mamba install -c conda-forge fpocket`).
Verified live: fpocket writes its output as
`<input_dir>/<input_stem>_out/<input_stem>_info.txt`, next to the input file
regardless of CWD — this script relies on that exact layout.

    PYTHONPATH=. python scripts/compute_pocket_descriptors.py --ids-file uniprot_ids.txt

Writes (merge, not clobber, one file per field — matching the existing
_SCALAR_FIELDS -> one-TSV-per-field convention in negaverse/io/annotations.py):
    local-docs/annotations/pocket_volume.tsv
    local-docs/annotations/pocket_hydrophobicity.tsv
    local-docs/annotations/pocket_polarity.tsv
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.compute_hydrophobicity import _sars_ids
from scripts.fetch_alphafold_structures import fetch_many

ANNOT_DIR = Path("local-docs/annotations")
FPOCKET_DIR = Path("local-docs/alphafold")   # fpocket writes next to the .cif it's given


def _parse_top_pocket(info_path: Path) -> dict[str, float] | None:
    """Parse fpocket's info.txt, return Pocket 1's fields as a flat dict, or
    None if the file has no pockets at all (a valid outcome — some proteins
    genuinely have no detectable cavity)."""
    text = info_path.read_text()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return None
    first = blocks[0].splitlines()
    fields: dict[str, float] = {}
    for line in first[1:]:                      # first line is "Pocket 1 :"
        if ":" not in line:
            continue
        key, _, val = line.rpartition(":")
        key, val = key.strip(), val.strip()
        try:
            fields[key] = float(val)
        except ValueError:
            continue
    return fields or None


def compute_one(accession: str, cif_path: Path) -> dict[str, float] | None:
    out_dir = cif_path.parent / f"{cif_path.stem}_out"
    info_path = out_dir / f"{cif_path.stem}_info.txt"
    if not info_path.exists():
        subprocess.run(["fpocket", "-f", str(cif_path)], check=False,
                        capture_output=True, cwd=str(cif_path.parent))
    if not info_path.exists():                   # fpocket found nothing, or errored
        return None
    top = _parse_top_pocket(info_path)
    if top is None:
        return None
    out = {}
    if "Volume" in top:
        out["pocket_volume"] = round(top["Volume"], 4)
    if "Hydrophobicity score" in top:
        out["pocket_hydrophobicity"] = round(top["Hydrophobicity score"], 4)
    if "Proportion of polar atoms" in top:
        out["pocket_polarity"] = round(top["Proportion of polar atoms"] / 100.0, 4)
    return out or None


def compute(accessions: list[str]) -> dict[str, dict[str, float]]:
    print(f"fetching AlphaFold structures for {len(accessions)} accessions...")
    af = fetch_many(accessions, out_dir=FPOCKET_DIR)
    results: dict[str, dict[str, float]] = {}
    n_confident = sum(1 for r in af.values() if not r["low_confidence"])
    print(f"{n_confident}/{len(accessions)} have a confident structure; running fpocket on those...")
    for acc, row in af.items():
        if row["low_confidence"]:
            continue
        desc = compute_one(acc, Path(row["cif_path"]))
        if desc is not None:
            results[acc] = desc
    print(f"{len(results)}/{n_confident} produced at least one pocket descriptor")
    return results


def _write_merged(path: Path, new_values: dict[str, float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                merged[k] = v
    merged.update({k: str(v) for k, v in new_values.items()})
    with open(path, "w") as fh:
        for n in sorted(merged):
            fh.write(f"{n}\t{merged[n]}\n")
    return len(merged)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compute_pocket_descriptors")
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

    results = compute(ids)
    for field, path in (
        ("pocket_volume", ANNOT_DIR / "pocket_volume.tsv"),
        ("pocket_hydrophobicity", ANNOT_DIR / "pocket_hydrophobicity.tsv"),
        ("pocket_polarity", ANNOT_DIR / "pocket_polarity.tsv"),
    ):
        values = {acc: d[field] for acc, d in results.items() if field in d}
        total = _write_merged(path, values)
        print(f"wrote {len(values)} new / {total} total values to {path}")


if __name__ == "__main__":
    main()
