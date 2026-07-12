"""Fill the SARS-CoV-2 viral proteins' hydrophobicity so the dashboard z-axis is real.

The `surface_hydrophobicity` annotation (local-docs/annotations/hydrophobicity.tsv)
covers human/host proteins but NOT the 29 SARS viral proteins — they aren't in
UniProt and have no AlphaFold model, so compute_surface_hydrophobicity.py skips
them. Every SARS positive/negative is a host–viral pair, so a missing viral value
collapses the whole "chemistry match" axis to z=0 (see negaverse/viz/interactive.py
hyd()). Here we add a sequence-level Kyte-Doolittle score (the same Tier-2 proxy
compute_hydrophobicity.py uses) for any graph node that has a local sequence but no
hydrophobicity value yet — host values are left untouched.

    PYTHONPATH=. python3 scripts/compute_sars_viral_hydrophobicity.py
"""
from __future__ import annotations
from pathlib import Path

from negaverse.io import load_sars_cov2_graph
from scripts.compute_hydrophobicity import _score        # normalized Kyte-Doolittle in [0,1]

OUT = Path("local-docs/annotations/hydrophobicity.tsv")
_SEQ = Path("local-docs/sars/sequences.tsv")


def _load_sequences() -> dict[str, str]:
    seqs = {}
    for line in _SEQ.read_text().splitlines():
        if "\t" in line:
            i, s = line.split("\t")[:2]
            seqs[i.strip()] = s.strip()
    return seqs


def main() -> None:
    have = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t")[:2]
                have[k] = v
    seqs = _load_sequences()
    nodes = set(load_sars_cov2_graph().g.nodes())

    added = 0
    for n in nodes:
        if n in have or n not in seqs:
            continue
        s = _score(seqs[n])
        if s is not None:
            have[n] = f"{s}"
            added += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(have.items())))
    covered = sum(1 for n in nodes if n in have)
    print(f"added {added} viral hydrophobicity scores; "
          f"SARS graph now {covered}/{len(nodes)} nodes covered; wrote {OUT}")


if __name__ == "__main__":
    main()
