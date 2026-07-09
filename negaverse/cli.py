"""End-to-end demo: SARS-CoV-2 interactome -> matched train/eval negatives.

    python -m negaverse.cli                 # run with defaults, write to out/
    python -m negaverse.cli --n-eval 500 --n-train 500
    python -m negaverse.cli --no-literature # skip the LLM pass explicitly

The LLM literature stream runs by default and is automatically skipped when no
API key (ANTHROPIC_API_KEY / OPENROUTER_API_KEY) is available. Writes:
    out/negatives.csv         one row per emitted negative (the output contract, §7)
    out/negatives.jsonl       same, with full nested provenance
    out/stats.json            pipeline + validation summary
    out/literature_cards.json LLM verdicts (when the literature stream runs)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import eval as ev
from .io import load_sars_cov2_graph, load_negatome_pairs
from .pipeline import PipelineConfig, run_pipeline
from .streams import build_filters, LiteratureFilter


def _load_dotenv(path: str | Path = ".env") -> None:
    """Best-effort .env loader (stdlib only) so credentials in .env are picked
    up without an extra dependency. Existing env vars are not overridden."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _collect_literature(records, out) -> dict:
    """Pull the gated LLM verdicts out of record provenance (the literature
    filter ran inside the pipeline), write the cards, and summarize."""
    cards, seen = [], set()
    for r in records:
        g = (r.provenance or {}).get("gated", {}).get("literature")
        if not g or (r.u, r.v) in seen:
            continue
        seen.add((r.u, r.v))
        cards.append({"u": r.u, "v": r.v, "verdict": g["verdict"],
                      "confidence": g["verdict_confidence"],
                      "votes": g.get("votes"), "agreement": g.get("agreement"),
                      "vote_counts": g.get("vote_counts"),
                      "rationale": g["rationale"], "evidence": g["evidence"],
                      "model": g["model"]})
    if not cards:
        return {"status": "skipped", "reason": "no cards (no key or no contested pairs)"}
    with open(out / "literature_cards.json", "w") as fh:
        json.dump(cards, fh, indent=2)
    from collections import Counter
    verdicts = Counter(c["verdict"] for c in cards)
    return {"status": "ran", "model": cards[0]["model"], "cards": len(cards),
            "verdicts": dict(verdicts), "file": str(out / "literature_cards.json")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--no-host-host", action="store_true",
                    help="drop host-host PPI edges (weakens the topological stream)")
    # literature-reasoning stream (gated Claude/OpenRouter cards, §5.2/§8.5).
    # ON by default; automatically skipped when no API key is available.
    ap.add_argument("--no-literature", action="store_true",
                    help="disable the LLM literature pass even if a key is available")
    ap.add_argument("--provider", choices=["auto", "anthropic", "openrouter"], default="auto",
                    help="LLM backend; 'auto' picks whichever key is present")
    ap.add_argument("--model", type=str, default=None, help="override the LLM model id")
    ap.add_argument("--literature-k", type=int, default=8,
                    help="how many contested pairs to send to the LLM")
    ap.add_argument("--votes", type=int, default=5,
                    help="best-of-N majority vote per pair (1 = single call)")
    args = ap.parse_args(argv)

    _load_dotenv()  # pick up ANTHROPIC_API_KEY / OPENROUTER_API_KEY from .env if present

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading SARS-CoV-2 interactome ...")
    graph = load_sars_cov2_graph(include_host_host=not args.no_host_host)
    print("  graph:", graph.summary(), "| edges:", graph.meta)

    cfg = PipelineConfig(n_eval=args.n_eval, n_train=args.n_train, seed=args.seed,
                         match_on_type="viral", gated_max=args.literature_k)

    # Build the hourglass filters; the gated literature filter runs the LLM
    # in-pipeline and is fused into confidence (skips itself without a key).
    filters = build_filters(cfg.modality, ["known_positive_veto", "structured", "embedding"])
    if not args.no_literature:
        filters.append(LiteratureFilter(enabled=True, provider=args.provider,
                                        model=args.model, votes=args.votes))
        print(f"Literature stream: enabled (provider={args.provider}, "
              f"up to {args.literature_k} contested pairs, best-of-{args.votes} vote)")

    print("Running pipeline (VETO funnel -> GRADED parallel -> GATED literature -> match/split) ...")
    result = run_pipeline(graph, cfg, filters=filters)

    # ---- validation ----
    eval_records = [r for r in result.records if r.mode == "eval"]
    validation = {
        "leakage_known_positive": ev.leakage(graph, result.records),
        "degree_match": ev.degree_match(graph, eval_records, match_type="viral", seed=args.seed),
        "hardness_split": ev.hardness_split(result.records),
    }
    # gold check (expected: space mismatch on the viral-host demo graph)
    try:
        gold = load_negatome_pairs()
        validation["gold_recall"] = ev.gold_recall(result.records, gold)
    except FileNotFoundError:
        validation["gold_recall"] = {"note": "negatome2 files not found"}

    # ---- gated literature verdicts (fused in-pipeline; extract cards) ----
    validation["literature"] = (
        {"status": "disabled"} if args.no_literature
        else _collect_literature(result.records, out))

    # ---- write outputs ----
    rows = [r.as_row() for r in result.records]
    df = pd.DataFrame(rows)
    csv_cols = [c for c in df.columns if c != "provenance"]
    df[csv_cols].to_csv(out / "negatives.csv", index=False)
    with open(out / "negatives.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"stats": result.stats, "validation": validation}
    with open(out / "stats.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- report ----
    print("\n=== pipeline stats ===")
    print(json.dumps(result.stats, indent=2))
    print("\n=== validation ===")
    print(json.dumps(validation, indent=2))
    print(f"\nwrote {len(rows)} negative records to {out}/negatives.csv, .jsonl")

    print("\n=== sample records ===")
    show = [c for c in ["u", "v", "mode", "confidence", "hardness",
                        "stream_structured", "stream_embedding", "flags"]
            if c in df.columns]
    print(df[show].sort_values("confidence", ascending=False).head(6).to_string(index=False))
    fn = df[df["flags"].str.contains("suspected_false_negative", na=False)]
    print(f"\nsuspected false negatives flagged: {len(fn)}")
    if len(fn):
        print(fn[show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
