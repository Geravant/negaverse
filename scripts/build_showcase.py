"""Build out/showcase.html — the single combined page for the jury / demo video.

    PYTHONPATH=. python3 scripts/build_showcase.py

Stitches the interactive 3D injection backtest together with the interactive 3D
pair maps from existing runs. Expects (regenerate first if missing):
  * out/interactive3d.json       — HuRI map   (python -m negaverse.viz --dataset huri)
  * out/sars/interactive3d.json  — SARS map + LLM verdicts (python -m negaverse.cli --out out/sars)
"""
from __future__ import annotations
from pathlib import Path
from negaverse.viz.showcase import build_showcase

_HURI_CAP = (
    "The human interactome (HuRI). All three axes are real here — every protein is annotated. "
    "<b>x</b> = looks like a real interaction (network shape); <b>y</b> = biology allows it "
    "(shared cell compartment); <b>z</b> = chemistry match (surface hydrophobicity). "
    "Colour = regime. Hover any dot for the two proteins and why it is flagged; "
    "real interactions cluster high-x, our chosen negatives sit low-x and clean, the "
    "<span style='color:#e63946'>risky</span> tail creeps toward the positive cloud.")

_SARS_CAP = (
    "The SARS-CoV-2 host–viral interactome — the transparency story. <b>x</b> = looks like a real "
    "interaction (network shape); <b>z</b> = chemistry match (surface hydrophobicity, computed for "
    "the viral proteins from sequence so the axis is real here too). Hover a "
    "<span style='color:#e63946'>risky</span> pair to read the <b>LLM literature verdict and "
    "reasoning inline</b> (the pairs the pipeline was unsure about and sent to literature review). "
    "<div class='note'><b>Note on the y-axis:</b> “biology allows it” is GO cell-compartment overlap, "
    "which needs both proteins annotated — but SARS viral proteins mostly aren’t (2 of 31). The random "
    "baseline here is <b>type-matched</b> (host–viral, like the positives and our negatives), so every "
    "regime sits at <i>y</i>≈0 and the compartment axis is uniformly uninformative on SARS — separation "
    "lives on <b>x</b> (network shape) and <b>z</b> (chemistry). Read the biology axis on the HuRI map "
    "above, where every protein is annotated.</div>")

_DRYAD_CAP = (
    "The DRYAD PPI benchmark — very sparse (avg degree ≈0.35), so network shape can’t "
    "separate pairs. Here <b>x</b> instead reads a protein-<i>sequence</i> (ESM2) model trained "
    "to tell real interactors from known non-interactors (AUROC ≈0.93); <b>y</b> = shared "
    "compartment, <b>z</b> = chemistry match. Hover a <span style='color:#e63946'>risky</span> "
    "pair for the LLM literature verdict — same transparency on a second, independent dataset.")

maps = [
    ("Interactive 3D map — HuRI", _HURI_CAP, "out/interactive3d.json"),
    ("Interactive 3D map — DRYAD", _DRYAD_CAP, "out/dryad/interactive3d.json"),
    ("Interactive 3D map — SARS-CoV-2", _SARS_CAP, "out/sars/interactive3d.json"),
]

missing = [p for _, _, p in maps if not Path(p).exists()]
if missing:
    print("WARNING: missing map json (panel will be skipped):")
    for m in missing:
        hint = ("python -m negaverse.viz --dataset huri" if "sars" not in m
                else "python -m negaverse.cli --out out/sars")
        print(f"  {m}   → generate with: {hint}")

out = build_showcase("out/showcase.html", maps)
print("wrote", out)
