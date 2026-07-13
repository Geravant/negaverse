"""Render the Phase-1 demo panels + a single-page HTML dashboard.

    python -m negaverse.viz                  # SARS-CoV-2 demo graph
    python -m negaverse.viz --dataset huri   # human PPI graph
    python -m negaverse.viz --dataset dryad  # DRYAD PPI benchmark (built from its positives)
    python -m negaverse.viz --dataset upna   # UPNA-PPI benchmark (built from its positives)

Writes out/*.png and out/report.html (open it in a browser).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..graph import TypedInteractionGraph
from ..pipeline import PipelineConfig, run_pipeline
from ..streams import build_filters, LiteratureFilter
from .. import eval as ev
from ..cli import _collect_literature
from ..io import load_sars_cov2_graph, load_huri_graph
from . import render_all, build_report

_MAP_DIR = Path("local-docs/mappings")


def _load_dotenv(path: str | Path = ".env") -> None:
    """Best-effort .env loader (stdlib) so ANTHROPIC/OPENROUTER keys are picked up."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _combined_name(symbol: str, synonyms: list[str], full_name: str) -> str:
    """One display string combining symbol + full protein name + synonyms, so
    the literature judge gets more to recognize the protein by and reason
    about its function than a bare symbol alone (e.g. "TP53 — Cellular tumor
    antigen p53 (aka P53, TRP53)")."""
    s = symbol
    if full_name:
        s += f" — {full_name}"
    if synonyms:
        s += f" (aka {', '.join(synonyms)})"
    return s


def _load_names(dataset: str) -> dict:
    """id -> human-readable gene identity (symbol + full name + synonyms) so
    the literature judge can reason. HuRI nodes are ENSG ids; DRYAD nodes are
    UniProt accessions (mapped via ENSG). Reads ensg_symbol.tsv's
    ensg<TAB>symbol<TAB>synonyms(comma-sep)<TAB>full_name format
    (scripts/build_ensg_symbol_map.py)."""
    ensg_name = {}
    f = _MAP_DIR / "ensg_symbol.tsv"
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip() or line.startswith("#") or "\t" not in line:
                continue
            parts = line.split("\t")
            ensg, sym = parts[0].strip(), parts[1].strip()
            syns = [s.strip() for s in parts[2].split(",")] if len(parts) > 2 and parts[2] else []
            full = parts[3].strip() if len(parts) > 3 else ""
            if sym:
                ensg_name[ensg] = _combined_name(sym, syns, full)
    if dataset == "huri":
        return ensg_name
    if dataset == "dryad":                       # UniProt -> (first) ENSG -> name
        names, up = {}, _MAP_DIR / "uniprot_ensg_human.tsv"
        if up.exists():
            for line in up.read_text().splitlines():
                if "\t" in line and not line.startswith("#"):
                    acc, ensgs = line.split("\t")[:2]
                    name = ensg_name.get(ensgs.split(",")[0].strip())
                    if name:
                        names[acc.strip()] = name
        return names
    if dataset == "upna":                         # UniProt -> original gene symbol
        return {acc: sym for sym, acc in _load_upna_symbol_to_uniprot().items()}
    return {}

_DRYAD_TSV = "local-docs/dryad-ppi/benchmarks/positives_and_negatives.tsv"
_DRYAD_ESM2_NPZ = "local-docs/dryad-ppi/esm2_t6_emb.npz"


def _load_dryad_graph() -> TypedInteractionGraph:
    """DRYAD ships as labelled positive/negative pairs, not a graph. Build the
    interaction graph from its positives so the same pipeline can run on it."""
    pos = []
    with open(_DRYAD_TSV) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                pos.append((a, b))
    nodes = {p for e in pos for p in e}
    return TypedInteractionGraph.from_edges(
        pos, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="dryad")


def _load_dryad_gold_negatives() -> list[tuple[str, str]]:
    """DRYAD's own labelled negative benchmark — plotted in place of
    freshly-generated random pairs (see plots.py::plot_quadrant)."""
    neg = []
    with open(_DRYAD_TSV) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            if cat == "negative":
                a, b = pair.split("_")
                neg.append((a, b))
    return neg


_UPNA_DIR = Path("local-docs/upna-ppi")


_UPNA_SYMBOL_MAP = Path("local-docs/mappings/gene_symbol_to_uniprot.tsv")


def _load_upna_symbol_to_uniprot() -> dict[str, str]:
    """gene symbol -> UniProt accession (scripts/build_gene_symbol_uniprot_map.py).
    A symbol with no reviewed human match is simply absent — see that script."""
    m: dict[str, str] = {}
    if _UPNA_SYMBOL_MAP.exists():
        for line in _UPNA_SYMBOL_MAP.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "\t" in line:
                sym, acc = line.split("\t")[:2]
                m[sym] = acc
    return m


def _load_upna_graph() -> TypedInteractionGraph:
    """UPNA-PPI ships as separate positive/negative CSVs (HGNC gene symbols),
    not a graph, and is dense (17,974 proteins, ~4.58M edges) — too large for
    the production TopologyFilter's per-pair set loop (see
    scripts/upna_topology_separation.py, which needed a separate scipy-sparse
    fast path just to score it). Restrict to the ~5,037-protein universe
    covered by their own topological negatives (TPPNI_*.csv) — the same
    scoping that script already uses, so the real production pipeline runs
    on this data directly. Uniformly typed "protein" (not split like SARS's
    viral/host) so rules/ppi.yaml's applies_to: [protein, protein] matches.

    Remapped to UniProt accessions (scripts/build_gene_symbol_uniprot_map.py)
    rather than kept as native gene symbols: this is what lets UPNA reuse
    every existing UniProt-space annotation/known-positive source (GO
    compartments, hydrophobicity, STRING, BioGRID) instead of needing its own
    gene-symbol-keyed copies of all of them. A symbol with no reviewed human
    UniProt match (~2% of the universe) is silently dropped — same
    silent-abstain convention as every other loader here. The literature
    judge still sees the original gene symbol via _load_names."""
    import pandas as pd

    def _pairs(pattern: str) -> list[tuple[str, str]]:
        out = []
        for f in sorted(_UPNA_DIR.glob(pattern)):
            for chunk in pd.read_csv(f, usecols=["SymbolA", "SymbolB"], chunksize=300_000):
                for a, b in zip(chunk["SymbolA"].astype(str), chunk["SymbolB"].astype(str)):
                    if a != b:
                        out.append((a, b))
        return out

    sym2acc = _load_upna_symbol_to_uniprot()
    universe_sym = {s for pair in _pairs("TPPNI_*.csv") for s in pair}
    universe = {sym2acc[s] for s in universe_sym if s in sym2acc}
    # a universe restriction scopes *which proteins are in the graph*, not just
    # which edges — otherwise a protein with no in-universe positive edge would
    # silently vanish instead of being an admissible (isolated) candidate node.
    pos = [(sym2acc[a], sym2acc[b]) for a, b in _pairs("PPI_part_*.csv")
           if a in universe_sym and b in universe_sym and a in sym2acc and b in sym2acc]
    return TypedInteractionGraph.from_edges(
        pos, {n: "protein" for n in universe},
        admissible_types=[("protein", "protein")], name="upna-ppi")


def _load_upna_gold_negatives() -> list[tuple[str, str]]:
    """UPNA's own headline hard-negative benchmark (TPPNI, contrastive-L3),
    remapped to UniProt like _load_upna_graph — plotted in place of
    freshly-generated random pairs (see plots.py::plot_quadrant)."""
    import pandas as pd
    sym2acc = _load_upna_symbol_to_uniprot()

    def _pairs(pattern: str) -> list[tuple[str, str]]:
        out = []
        for f in sorted(_UPNA_DIR.glob(pattern)):
            for chunk in pd.read_csv(f, usecols=["SymbolA", "SymbolB"], chunksize=300_000):
                for a, b in zip(chunk["SymbolA"].astype(str), chunk["SymbolB"].astype(str)):
                    if a != b:
                        out.append((a, b))
        return out

    tppni = _pairs("TPPNI_*.csv")
    universe_sym = {s for pair in tppni for s in pair}
    return [(sym2acc[a], sym2acc[b]) for a, b in tppni
            if a in universe_sym and b in universe_sym and a in sym2acc and b in sym2acc]


def _dryad_sequence_axis(graph, seed):
    """A supervised "looks like a real interaction" x-axis for DRYAD.

    Topology is inert here (the graph is too sparse) and ESM2 manifold-resemblance
    is flat (AUROC ~0.57). But an ESM2 classifier trained on positives vs DRYAD's
    GOLD non-interactors (concat features, RandomForest) separates at AUROC ~0.93,
    so read its P(real) as x. Leakage-free: positives are scored out-of-fold; the
    other regimes plotted on the map (random, our chosen negatives, risky) were
    never in training and were selected by topology — independent of ESM2 — so
    scoring them here is honest, and reveals whether any "look real" by sequence.
    Returns (x_fn, title, missing_note) for compute_traces; None where unbuildable."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from ..io.embeddings import load_embeddings_npz

    try:
        emb = load_embeddings_npz(_DRYAD_ESM2_NPZ)
    except Exception:
        return None

    def _f(u, v):
        if u not in emb or v not in emb:
            return None
        a, b = np.asarray(emb[u], float), np.asarray(emb[v], float)
        return np.concatenate([np.minimum(a, b), np.maximum(a, b)])   # order-invariant concat

    pos = [tuple(e) for e in graph.g.edges()]
    gold = []
    for line in Path(_DRYAD_TSV).read_text().splitlines()[1:]:
        pr, cat = line.split("\t")
        if cat == "negative":
            a, b = pr.split("_"); gold.append((a, b))

    X, y, keys = [], [], []
    for e, lab in [(e, 1) for e in pos] + [(e, 0) for e in gold]:
        f = _f(*e)
        if f is not None:
            X.append(f); y.append(lab); keys.append(frozenset(e))
    if len(set(y)) < 2:
        return None
    X, y = np.asarray(X), np.asarray(y)
    oof = cross_val_predict(RandomForestClassifier(200, random_state=seed, n_jobs=-1),
                            X, y, cv=5, method="predict_proba")[:, 1]
    oof_map = {k: float(s) for k, s in zip(keys, oof)}        # training pairs -> OOF score
    full = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X, y)

    def x_fn(u, v):
        k = frozenset((u, v))
        if k in oof_map:
            return oof_map[k]                                 # positive/gold -> out-of-fold
        f = _f(u, v)
        return None if f is None else float(full.predict_proba([f])[0, 1])

    return (x_fn, "looks real (ESM2 model vs gold non-interactors)", "no sequence embedding")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse.viz")
    ap.add_argument("--dataset", choices=["sars", "huri", "dryad", "upna"], default="sars")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-selection", default="stacked",
                    choices=["hard", "safe", "stacked", "mixture", "psm"],
                    help="how the emitted training negatives are chosen (default stacked)")
    ap.add_argument("--no-literature", action="store_true",
                    help="skip the LLM literature review of risky pairs (on by default when a key is present)")
    ap.add_argument("--votes", type=int, default=3,
                    help="best-of-N majority vote per pair in the literature review")
    ap.add_argument("--literature-k", type=int, default=60,
                    help="max risky pairs sent to the LLM (bounds cost)")
    args = ap.parse_args(argv)
    ts = args.train_selection
    _load_dotenv()

    _names = ["known_positive_veto", "structured", "topology", "rules"]
    if args.dataset == "huri":
        graph = load_huri_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=args.n_train, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed, train_selection=ts,
                             filters=_names, gated_max=args.literature_k)
    elif args.dataset == "dryad":
        graph = _load_dryad_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=args.n_train, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed, train_selection=ts,
                             filters=_names, gated_max=args.literature_k)
    elif args.dataset == "upna":
        graph = _load_upna_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=args.n_train, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed, train_selection=ts,
                             filters=_names, gated_max=args.literature_k)
    else:
        graph = load_sars_cov2_graph()
        cfg = PipelineConfig(n_eval=args.n_train, n_train=args.n_train, seed=args.seed,
                             match_on_type="viral", train_selection=ts,
                             filters=_names, gated_max=args.literature_k)

    # Build filter instances so we can attach the gated literature reviewer with a
    # dataset-specific gene-symbol map (so the judge reasons about named genes, not raw ids).
    filters = build_filters(cfg.modality, _names)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    if not args.no_literature and has_key:
        names = _load_names(args.dataset)
        filters.append(LiteratureFilter(enabled=True, votes=args.votes, names=names))
        print(f"literature review: enabled (best-of-{args.votes}, {len(names)} gene symbols mapped)")
    elif not args.no_literature:
        print("literature review: skipped (no ANTHROPIC_API_KEY / OPENROUTER_API_KEY)")

    print(f"graph: {graph.summary()}")
    result = run_pipeline(graph, cfg, filters=filters)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    validation = {
        "leakage_known_positive": ev.leakage(graph, result.records),
        "hardness_split": ev.hardness_split(result.records),
        "literature": ({"status": "disabled"} if args.no_literature
                       else _collect_literature(result.records, out)),
    }
    (out / "stats.json").write_text(json.dumps(
        {"stats": result.stats, "validation": validation}, indent=2))

    # DRYAD's graph is too sparse for topology to separate anything (only ~0.2%
    # of non-edges share a neighbour → every candidate sits at the x-floor and our
    # negatives land on top of random), and ESM2 manifold-*resemblance* is flat
    # too (AUROC ~0.57). What *does* separate DRYAD is a supervised ESM2 model
    # trained on positives vs the GOLD non-interactors (AUROC ~0.93). Use its
    # P(real) as the x-axis (see _dryad_sequence_axis).
    x_axis = _dryad_sequence_axis(graph, args.seed) if args.dataset == "dryad" else None

    # Datasets that ship their own labelled negative benchmark get it plotted
    # instead of freshly-generated random pairs (see plots.py::plot_quadrant)
    # — a real external negative set is a more meaningful comparison than a
    # synthetic one when one exists.
    gold_negatives = None
    if args.dataset == "dryad":
        gold_negatives = _load_dryad_gold_negatives()
    elif args.dataset == "upna":
        gold_negatives = _load_upna_gold_negatives()
    if gold_negatives is not None:
        # DRYAD's gold negatives come from the FULL labelled TSV, which
        # includes proteins that appear only in negative-labelled pairs —
        # _load_dryad_graph builds nodes from positives only, so such a
        # protein is never a graph node at all. Filter to pairs the graph
        # actually knows about (silent-abstain, same convention as every
        # other loader here) — an unfiltered pair here would KeyError deep
        # inside the topology/manifold plots' adjacency lookups.
        graph_nodes = set(graph.g.nodes())
        gold_negatives = [(a, b) for a, b in gold_negatives if a in graph_nodes and b in graph_nodes]

    render_all(graph, result.records, out, stats=result.stats, seed=args.seed, x_axis=x_axis,
              gold_negatives=gold_negatives)
    report = build_report(out, title="negaverse", subtitle=f"{args.dataset} demo run")
    print(f"wrote dashboard: {report}")


if __name__ == "__main__":
    main()
