"""Build a per-gene multi-species alignment from the ortholog CDS sequences
scripts/fetch_orthologs.py cached, for scripts/estimate_phangorn_trees.R
(RERconverge's own recommended tree-building method: fixed master-topology
branch-length estimation via phangorn, not free per-gene ML tree search).

Per gene: write a FASTA of every cached species' CDS (human + whichever
SPECIES_PANEL orthologs were found), align with **MAFFT** (verified
installable, `osx-arm64`, conda-forge).

**Why not build the tree here too (this script used to run FastTree per
gene):** free per-gene ML tree search (FastTree) frequently disagrees with
the master tree's topology on single-locus, ~15-species alignments —
RERconverge's readTrees() discards any gene whose topology conflicts with
the master tree ("discordant tree topology... returning NA row"), and
empirically this discarded 18/19 genes in an early smoke test. RERconverge's
own PhangornTreeBuildingWalkthrough.Rmd recommends the opposite approach for
exactly this scale: fix the topology to a known master tree and only
estimate branch lengths per gene (via phangorn's `pml`/`optim.pml`, wrapped
by RERconverge's `estimatePhangornTreeAll`) — this avoids discordance by
construction, since every gene tree then shares the same topology and only
differs in relative branch lengths, which is exactly the RER signal we want.
scripts/estimate_phangorn_trees.R does that step, using
scripts/data/vertebrate_master_tree.nwk as the fixed topology.

Genes with fewer than MIN_SPECIES cached orthologs (+human) are skipped
entirely (no alignment from 2-3 sequences) — this is the same kind of
minimum-data gate as fpocket's "no confident structure -> no score".

    PYTHONPATH=. python scripts/build_gene_alignments.py --ids-file ensembl_ids.txt

Writes (gitignored):
    local-docs/alignments/<ensembl_gene_id>.aligned.fasta
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ORTHOLOG_DIR = Path("local-docs/orthologs")
OUT_DIR = Path("local-docs/alignments")
RAW_DIR = Path("local-docs/alignments_raw")   # pre-alignment scratch; kept out of
                                               # OUT_DIR so estimatePhangornTreeAll's
                                               # alndir scan only sees final alignments
MIN_SPECIES = 6          # need enough taxa for a branch-length estimate to mean anything


def _write_fasta(species_seqs: dict[str, str], path: Path) -> None:
    with open(path, "w") as fh:
        for species, seq in species_seqs.items():
            fh.write(f">{species}\n{seq}\n")


def build_one(ensembl_id: str, ortholog_dir: Path = ORTHOLOG_DIR,
              out_dir: Path = OUT_DIR) -> Path | None:
    """Returns the alignment's path, or None if skipped (too few species, or
    MAFFT failed)."""
    # bare "<gene_id>.fasta" — no ".aligned" in the name, since alignment
    # status is conveyed by directory (out_dir vs RAW_DIR), not filename;
    # estimatePhangornTreeAll derives its gene names from the filename minus
    # only the final ".fasta" extension, so any extra suffix here would leak
    # into the RER matrix's gene IDs.
    aligned_path = out_dir / f"{ensembl_id}.fasta"
    if aligned_path.exists():
        return aligned_path

    ortholog_path = ortholog_dir / f"{ensembl_id}.json"
    if not ortholog_path.exists():
        return None
    import json
    data = json.loads(ortholog_path.read_text())
    species_seqs = {species: row["cds"] for species, row in data.items() if row.get("cds")}
    if len(species_seqs) < MIN_SPECIES:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = RAW_DIR / f"{ensembl_id}.fasta"
    _write_fasta(species_seqs, fasta_path)

    with open(aligned_path, "w") as out_fh:
        r = subprocess.run(["mafft", "--auto", "--quiet", str(fasta_path)],
                            stdout=out_fh, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or aligned_path.stat().st_size == 0:
        aligned_path.unlink(missing_ok=True)
        return None
    return aligned_path


def build_many(ensembl_ids: list[str], ortholog_dir: Path = ORTHOLOG_DIR,
               out_dir: Path = OUT_DIR) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for i, gid in enumerate(ensembl_ids, 1):
        path = build_one(gid, ortholog_dir, out_dir)
        if path is not None:
            results[gid] = path
        if i % 25 == 0 or i == len(ensembl_ids):
            print(f"  aligned {len(results)}/{i} genes ({i}/{len(ensembl_ids)} attempted)")
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="build_gene_alignments")
    ap.add_argument("--ids-file", required=True, help="Ensembl gene IDs, one per line")
    args = ap.parse_args(argv)
    ids = sorted(set(l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()))
    print(f"{len(ids)} Ensembl gene IDs")

    results = build_many(ids)
    print(f"aligned {len(results)}/{len(ids)} genes "
          f"(skipped: no cached orthologs, <{MIN_SPECIES} species, or MAFFT failed)")


if __name__ == "__main__":
    main()
