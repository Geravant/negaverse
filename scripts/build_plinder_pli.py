"""Build the PLI (protein-ligand) data foundation from PLINDER.

PLINDER (https://plinder.sh) is a gold-standard protein-ligand interaction set.
Its annotation table lives in a PUBLIC GCS bucket, and parquet is columnar, so we
read ONLY the ~dozen columns we need over anonymous range-reads — no gigabyte
download, no 3D structures.

Writes (all gitignored under local-docs/):
  * plinder/pli_edges.tsv                 protein(UniProt) <TAB> ligand(CCD) binders
  * annotations/plinder_ligand_logp.tsv   CCD  -> Crippen clogP        (field: logp)
  * annotations/plinder_ligand_volume.tsv CCD  -> heavy-atom count     (field: volume)
  * annotations/plinder_ligand_tpsa.tsv   CCD  -> topological PSA       (field: tpsa)
  * annotations/plinder_pocket_volume.tsv UniProt -> median #pocket res (field: pocket_volume)

A protein-ligand edge = a documented binder. Ligand properties are molecule-level
constants (dedupe by CCD); a protein's pocket size varies by system, so aggregate
(median) per UniProt. Ions and crystallization artifacts are dropped.

    PYTHONPATH=. python scripts/build_plinder_pli.py [--max-rows N]

Requires pyarrow (already a dep via the benchmark). Network: anonymous GCS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pyarrow.fs import GcsFileSystem

BUCKET = "plinder/2024-06/v2/index/annotation_table.parquet"
OUT_EDGES = Path("local-docs/plinder/pli_edges.tsv")
ANN = Path("local-docs/annotations")
COLS = ["system_pocket_UniProt", "ligand_ccd_code", "ligand_is_ion", "ligand_is_artifact",
        "ligand_crippen_clogp", "ligand_num_heavy_atoms", "ligand_tpsa",
        "ligand_molecular_weight", "ligand_num_rings", "ligand_qed",
        "system_num_pocket_residues"]


def _write_scalar(path: Path, d: dict[str, float], header: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {header}\n" + "".join(f"{k}\t{v}\n" for k, v in sorted(d.items())))
    return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows read (0 = all)")
    args = ap.parse_args()

    print(f"Reading {len(COLS)} columns from gs://{BUCKET} (anonymous) ...")
    fs = GcsFileSystem(anonymous=True)
    pf = pq.ParquetFile(fs.open_input_file(BUCKET))
    frames = []
    seen = 0
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg, columns=COLS).to_pandas()
        frames.append(df)
        seen += len(df)
        print(f"  row group {rg}: {len(df):,} rows (total {seen:,})")
        if args.max_rows and seen >= args.max_rows:
            break
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    if args.max_rows:
        df = df.iloc[:args.max_rows]
    print(f"  loaded {len(df):,} system rows")

    # keep real protein-ligand binders: valid UniProt + CCD, drop ions/artifacts
    df = df[df["system_pocket_UniProt"].notna() & df["ligand_ccd_code"].notna()]
    df = df[~df["ligand_is_ion"].fillna(False) & ~df["ligand_is_artifact"].fillna(False)]
    # a UniProt cell can list several chains ("P1;P2"); take the first as the pocket protein
    df["prot"] = df["system_pocket_UniProt"].astype(str).str.split(r"[;,\s]").str[0]
    df["lig"] = df["ligand_ccd_code"].astype(str)
    df = df[(df["prot"] != "") & (df["prot"] != "nan")]
    print(f"  after filtering ions/artifacts/missing: {len(df):,} rows")

    edges = df[["prot", "lig"]].drop_duplicates()
    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    OUT_EDGES.write_text("# protein(UniProt)<TAB>ligand(CCD) binders — PLINDER 2024-06/v2\n"
                         + "".join(f"{p}\t{l}\n" for p, l in edges.itertuples(index=False)))
    print(f"\nwrote {len(edges):,} unique protein-ligand edges "
          f"({edges['prot'].nunique():,} proteins x {edges['lig'].nunique():,} ligands) -> {OUT_EDGES}")

    # ligand properties (molecule-level constants) — dedupe by CCD
    lig = df.drop_duplicates("lig").set_index("lig")
    n1 = _write_scalar(ANN / "plinder_ligand_logp.tsv",
                       lig["ligand_crippen_clogp"].dropna().round(4).to_dict(),
                       "CCD -> Crippen clogP (PLINDER)")
    n2 = _write_scalar(ANN / "plinder_ligand_volume.tsv",
                       lig["ligand_num_heavy_atoms"].dropna().astype(int).to_dict(),
                       "CCD -> heavy-atom count (ligand size proxy, PLINDER)")
    n3 = _write_scalar(ANN / "plinder_ligand_tpsa.tsv",
                       lig["ligand_tpsa"].dropna().round(2).to_dict(),
                       "CCD -> topological polar surface area (PLINDER)")
    n5 = _write_scalar(ANN / "plinder_ligand_mw.tsv",
                       lig["ligand_molecular_weight"].dropna().round(2).to_dict(),
                       "CCD -> molecular weight (PLINDER)")
    n6 = _write_scalar(ANN / "plinder_ligand_rings.tsv",
                       lig["ligand_num_rings"].dropna().astype(int).to_dict(),
                       "CCD -> ring count (PLINDER)")
    n7 = _write_scalar(ANN / "plinder_ligand_qed.tsv",
                       lig["ligand_qed"].dropna().round(4).to_dict(),
                       "CCD -> QED drug-likeness (PLINDER)")
    # protein pocket size — median #pocket residues over that protein's systems
    pv = df.groupby("prot")["system_num_pocket_residues"].median().dropna().round(1).to_dict()
    n4 = _write_scalar(ANN / "plinder_pocket_volume.tsv", pv,
                       "UniProt -> median #pocket residues (pocket size proxy, PLINDER)")
    print(f"annotations: logp={n1:,} volume={n2:,} tpsa={n3:,} mw={n5:,} rings={n6:,} "
          f"qed={n7:,} pocket_volume={n4:,}")


if __name__ == "__main__":
    main()
