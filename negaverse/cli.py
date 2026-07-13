"""End-to-end demo: interactome -> matched train/eval negatives.

    python -m negaverse.cli                 # SARS-CoV-2 demo graph, write to out/
    python -m negaverse.cli --n-eval 500 --n-train 500
    python -m negaverse.cli --no-literature # skip the LLM pass explicitly
    python -m negaverse.cli run --input positives.tsv --modality ppi
                                             # run on any positives file instead
                                             # of the built-in SARS-CoV-2 graph

`run` is accepted as an optional leading subcommand (so `negaverse run ...`
works once the package is installed and exposes the `negaverse` console
script) but is not required — bare `python -m negaverse.cli ...` still works
for the built-in demo. --input takes a tab-separated positives file, one pair
per line: `u\tv` (both endpoints typed "protein", e.g. HuRI) or `u\tv\tu_type
\tv_type` (explicit per-endpoint types, e.g. viral/host — see
negaverse/io/generic.py). Without --input, the built-in SARS-CoV-2 demo graph
is used, unchanged from before.

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
import sys
from pathlib import Path

import pandas as pd

from . import eval as ev
from .io import load_sars_cov2_graph, load_negatome_pairs, load_generic_graph
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


def _load_graph(args):
    """Built-in SARS-CoV-2 demo graph, or a generic --input positives file."""
    if args.input:
        print(f"Loading graph from {args.input} ...")
        return load_generic_graph(args.input)
    print("Loading SARS-CoV-2 interactome ...")
    return load_sars_cov2_graph(include_host_host=not args.no_host_host)


def _flags_of(rec: dict) -> list[str]:
    fl = rec.get("flags")
    return fl.split(";") if isinstance(fl, str) else list(fl or [])


def _judge_remaining(args, out: Path) -> None:
    """Incremental pass: load a prior --out run and send its still-unjudged risky
    pairs (flagged suspected_false_negative, no LLM verdict yet) to the judge,
    up to --literature-k more. Re-fuses their confidence, updates the cards,
    negatives files, stats coverage, and (best-effort) the dashboard — so you can
    keep judging the tail across several invocations."""
    from .pipeline import PipelineConfig, _fuse_confidence
    from .streams import build_filters, LiteratureFilter

    jsonl = out / "negatives.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"--judge-remaining: no prior run at {jsonl} (run without it first)")
    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]

    cards_path = out / "literature_cards.json"
    cards = json.loads(cards_path.read_text()) if cards_path.exists() else []
    judged = {frozenset((c["u"], c["v"])) for c in cards}

    remaining = [r for r in recs if "suspected_false_negative" in _flags_of(r)
                 and frozenset((r["u"], r["v"])) not in judged]
    if not remaining:
        print("--judge-remaining: every risky pair already has a verdict — nothing to do.")
        return
    todo = remaining[:args.literature_k]
    print(f"--judge-remaining: {len(remaining)} unjudged risky pairs; "
          f"judging {len(todo)} (--literature-k {args.literature_k}) ...")

    graph = _load_graph(args)
    cfg = PipelineConfig(modality=args.modality,
                         match_on_type="viral" if not args.input else None)
    graded = build_filters(cfg.modality, ["structured", "topology"])
    lit = LiteratureFilter(enabled=True, provider=args.provider, model=args.model, votes=args.votes)
    for f in graded + [lit]:
        f.fit(graph)

    by_pair = {frozenset((r["u"], r["v"])): r for r in recs}
    new_cards = []
    for r in todo:
        u, v = r["u"], r["v"]
        sub, rep = {}, {}
        for f in graded:
            sc = f.score(graph, u, v)
            if sc.value is not None:
                sub[f.name] = sc.value
                rep[f.name] = (sc.evidence or {}).get("confidence")
        ls = lit.score(graph, u, v)
        ev = ls.evidence or {}
        if ev.get("gated_status") != "reviewed":
            print(f"  {u} x {v}: judge abstained ({ev.get('reason', ev.get('verdict'))})")
            continue
        if ls.value is not None:
            sub[lit.name] = ls.value
            rep[lit.name] = ev.get("confidence")
        conf = _fuse_confidence(sub, cfg.weights, cfg.fusion_mode, cfg.fusion_lam, rep)
        rr = by_pair[frozenset((u, v))]
        rr["stream_literature"] = ls.value
        rr["confidence"] = conf
        fl = _flags_of(rr)
        tag = "llm_" + ev["verdict"]
        if tag not in fl:
            fl.append(tag)
        rr["flags"] = ";".join(fl)
        new_cards.append({"u": u, "v": v, "verdict": ev["verdict"],
                          "confidence": ev["verdict_confidence"], "votes": ev.get("votes"),
                          "agreement": ev.get("agreement"), "vote_counts": ev.get("vote_counts"),
                          "rationale": ev["rationale"], "evidence": ev["evidence"],
                          "model": ev["model"]})
        print(f"  {u} x {v} -> {ev['verdict']} (conf {ev['verdict_confidence']})")

    cards_path.write_text(json.dumps(cards + new_cards, indent=2))
    df = pd.DataFrame(recs)
    df[[c for c in df.columns if c != "provenance"]].to_csv(out / "negatives.csv", index=False)
    with open(jsonl, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    stats_path = out / "stats.json"
    if stats_path.exists():
        summary = json.loads(stats_path.read_text())
        rc = summary.get("stats", {}).get("risky_coverage")
        if rc:
            rc["judged"] = rc.get("judged", 0) + len(new_cards)
            rc["unjudged"] = max(0, rc.get("risky", 0) - rc["judged"])
        stats_path.write_text(json.dumps(summary, indent=2))

    print(f"\njudged {len(new_cards)} more; literature_cards.json now holds "
          f"{len(cards) + len(new_cards)} verdicts. "
          f"{len(remaining) - len(todo)} risky pairs still unjudged.")
    if not args.no_report:
        try:
            from .viz import build_report
            build_report(out, title="negaverse", subtitle=f"{graph.name} run (judge-remaining)")
            print("refreshed dashboard: " + str(out / "report.html"))
        except Exception as e:
            print(f"(skipped dashboard refresh: {e})")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        argv = argv[1:]   # `negaverse run ...` == `negaverse ...` (see module docstring)

    ap = argparse.ArgumentParser(prog="negaverse run")
    ap.add_argument("--input", type=str, default=None,
                    help="tab-separated positives file (u,v[,u_type,v_type]); "
                         "replaces the built-in SARS-CoV-2 demo graph when given")
    ap.add_argument("--modality", type=str, default="ppi",
                    help="rule set to apply (see rules/<modality>.yaml); only used "
                         "with --input, since the demo graph is always ppi")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--no-host-host", action="store_true",
                    help="drop host-host PPI edges (weakens the topological stream); "
                         "only applies to the built-in SARS-CoV-2 demo graph")
    # literature-reasoning stream (gated Claude/OpenRouter cards, §5.2/§8.5).
    # ON by default; automatically skipped when no API key is available.
    ap.add_argument("--no-literature", action="store_true",
                    help="disable the LLM literature pass even if a key is available")
    ap.add_argument("--provider", choices=["auto", "anthropic", "openrouter"], default="auto",
                    help="LLM backend; 'auto' picks whichever key is present")
    ap.add_argument("--model", type=str, default=None, help="override the LLM model id")
    ap.add_argument("--literature-k", type=int, default=40,
                    help="max contested/risky pairs to send to the LLM per run "
                         "(the risky tail beyond this stays unjudged; see "
                         "risky_coverage in stats and --judge-remaining)")
    ap.add_argument("--judge-remaining", action="store_true",
                    help="don't run the full pipeline — load a prior --out run and "
                         "send its still-unjudged risky pairs (suspected_false_negative "
                         "without an LLM verdict) to the judge, up to --literature-k more")
    ap.add_argument("--no-report", action="store_true",
                    help="skip writing the out/report.html dashboard")
    ap.add_argument("--votes", type=int, default=5,
                    help="best-of-N majority vote per pair (1 = single call)")
    args = ap.parse_args(argv)

    _load_dotenv()  # pick up ANTHROPIC_API_KEY / OPENROUTER_API_KEY from .env if present

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.judge_remaining:
        _judge_remaining(args, out)
        return

    graph = _load_graph(args)
    print("  graph:", graph.summary(), "| edges:", graph.meta)

    cfg = PipelineConfig(n_eval=args.n_eval, n_train=args.n_train, seed=args.seed,
                         modality=args.modality, gated_max=args.literature_k,
                         match_on_type="viral" if not args.input else None)

    # Build the hourglass filters; the gated literature filter runs the LLM
    # in-pipeline and is fused into confidence (skips itself without a key).
    filters = build_filters(cfg.modality, ["known_positive_veto", "structured", "topology"])
    if not args.no_literature:
        filters.append(LiteratureFilter(enabled=True, provider=args.provider,
                                        model=args.model, votes=args.votes))
        print(f"Literature stream: enabled (provider={args.provider}, "
              f"up to {args.literature_k} contested pairs, best-of-{args.votes} vote)")

    print("Running pipeline (VETO funnel -> GRADED parallel -> GATED literature -> match/split) ...")
    result = run_pipeline(graph, cfg, filters=filters)

    # ---- validation ----
    eval_records = [r for r in result.records if r.mode == "eval"]
    # degree_match assumes viral/host node typing (the SARS-CoV-2 demo graph, or
    # any --input file that carries those two types); anything else has no
    # cross-type confounder for it to check.
    if {"viral", "host"} <= set(graph.node_type.values()):
        degree_match = ev.degree_match(graph, eval_records, match_type="viral", seed=args.seed)
    else:
        degree_match = {"note": "degree_match assumes viral/host node typing; "
                                "skipped for this graph's type space"}
    validation = {
        "leakage_known_positive": ev.leakage(graph, result.records),
        "degree_match": degree_match,
        "hardness_split": ev.hardness_split(result.records),
    }
    # gold check against Negatome (UniProt human-human); expect a space mismatch
    # note unless the input graph happens to share that ID space
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

    # ---- dashboard (best-effort; needs the viz extra) ----
    if not args.no_report:
        try:
            from .viz import render_all, build_report
            render_all(graph, result.records, out, stats=result.stats, seed=args.seed)
            report = build_report(out, title="negaverse",
                                  subtitle=f"{graph.name} run")
            print(f"\nwrote dashboard: {report}  (open in a browser)")
        except Exception as e:                       # matplotlib/sklearn absent, etc.
            print(f"\n(skipped dashboard: {e}; install with `pip install -e \".[viz]\"`)")

    # ---- report ----
    print("\n=== pipeline stats ===")
    print(json.dumps(result.stats, indent=2))
    print("\n=== validation ===")
    print(json.dumps(validation, indent=2))
    print(f"\nwrote {len(rows)} negative records to {out}/negatives.csv, .jsonl")

    print("\n=== sample records ===")
    show = [c for c in ["u", "v", "mode", "confidence", "hardness",
                        "stream_structured", "stream_topology", "flags"]
            if c in df.columns]
    print(df[show].sort_values("confidence", ascending=False).head(6).to_string(index=False))
    fn = df[df["flags"].str.contains("suspected_false_negative", na=False)]
    print(f"\nsuspected false negatives flagged: {len(fn)}")
    rc = result.stats.get("risky_coverage", {})
    if rc.get("unjudged"):
        print(f"  LLM-judged {rc['judged']} of {rc['risky']} risky pairs — "
              f"{rc['unjudged']} unjudged (raise --literature-k, now {rc.get('gated_cap')}, "
              f"or run: python -m negaverse.cli --judge-remaining --out {out})")
    if len(fn):
        print(fn[show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
