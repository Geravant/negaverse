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
from .graph import TypedInteractionGraph
from .io import load_sars_cov2_graph, load_negatome_pairs
from .pipeline import PipelineConfig, PipelineResult, run_pipeline


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


_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def _resolve_provider(requested: str) -> str | None:
    """Pick a backend whose key is present. 'auto' prefers Anthropic, then
    OpenRouter; an explicit provider is honoured only if its key exists.
    Returns None when no usable key is available (→ literature is skipped)."""
    if requested == "auto":
        for provider, env in _PROVIDER_ENV.items():
            if os.getenv(env):
                return provider
        return None
    return requested if os.getenv(_PROVIDER_ENV[requested]) else None


def _run_literature(graph: TypedInteractionGraph, result: PipelineResult,
                    provider: str, model: str | None, k: int, out) -> dict:
    """Gated LLM pass: judge the top-k most contested emitted negatives.
    Returns a summary; writes full cards to out/literature_cards.json.
    Never fatal — a missing key or API error degrades to a skip note."""
    resolved = _resolve_provider(provider)
    if resolved is None:
        want = "a provider key" if provider == "auto" else f"{_PROVIDER_ENV[provider]}"
        print(f"\nLiterature stream skipped: no API key found (set {want} in .env).")
        return {"status": "skipped", "reason": "no_api_key"}
    provider = resolved

    from .llm import LLMConfig, LLMController, LiteratureReasoner, LLMError

    # contest = flagged suspected FN or near-boundary, ranked by low confidence
    contested = sorted(
        [r for r in result.records
         if "suspected_false_negative" in r.flags or "near_boundary" in r.flags],
        key=lambda r: r.confidence,
    )[:k]
    if not contested:
        return {"status": "no_contested_pairs"}

    try:
        controller = LLMController(LLMConfig(provider=provider, model=model))
        reasoner = LiteratureReasoner(controller)
        print(f"\nRunning literature stream on {len(contested)} contested pairs via {controller.describe} ...")
        pairs = []
        for r in contested:
            ctx = {
                "u_type": graph.node_type.get(r.u),
                "v_type": graph.node_type.get(r.v),
                "u_degree": graph.degree(r.u),
                "v_degree": graph.degree(r.v),
                "embedding_link_score": r.streams.get("embedding"),
                "hardness_percentile": r.hardness,
            }
            pairs.append((r.u, r.v, ctx))
        cards = reasoner.reason_batch(pairs)
    except LLMError as e:
        print(f"  literature stream skipped: {e}")
        return {"status": "skipped", "reason": str(e)}

    payload = [c.as_dict() for c in cards]
    with open(out / "literature_cards.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    from collections import Counter
    verdicts = Counter(c.verdict for c in cards)
    return {
        "status": "ran",
        "model": controller.describe,
        "cards": len(cards),
        "verdicts": dict(verdicts),
        "file": str(out / "literature_cards.json"),
    }


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
    args = ap.parse_args(argv)

    _load_dotenv()  # pick up ANTHROPIC_API_KEY / OPENROUTER_API_KEY from .env if present

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading SARS-CoV-2 interactome ...")
    graph = load_sars_cov2_graph(include_host_host=not args.no_host_host)
    print("  graph:", graph.summary(), "| edges:", graph.meta)

    cfg = PipelineConfig(n_eval=args.n_eval, n_train=args.n_train, seed=args.seed,
                         match_on_type="viral")
    print("Running pipeline (candidates -> exclude -> 3-stream score -> fuse -> match/split) ...")
    result = run_pipeline(graph, cfg)

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

    # ---- gated literature-reasoning pass (§5.2), on by default ----
    if not args.no_literature:
        validation["literature"] = _run_literature(
            graph, result, provider=args.provider, model=args.model, k=args.literature_k,
            out=out)

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
