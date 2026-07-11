"""Literature-reasoning filter (ARCHITECTURE.md §5.2) — GATED stage.

Runs the LLM (Anthropic API or OpenRouter, via `negaverse.llm`) on the contested
pairs the pipeline routes to the gated stage, and maps the structured verdict
into the fused confidence:

  * safe_negative           -> high true-negative confidence (= verdict conf)
  * suspected_false_negative -> low  true-negative confidence (= 1 - verdict conf)
                                + a `suspected_false_negative` flag
  * uncertain               -> abstain (no confidence contribution)

Safe by default: `enabled=False` abstains (pure stub, no API calls) so library
use and tests never hit the network. The CLI constructs it with `enabled=True`.
Degrades gracefully — no key / API error -> abstain, pipeline unaffected.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register

_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
_DEFAULT_CACHE = "local-docs/cache/llm_verdicts.jsonl"


@register
class LiteratureFilter(Filter):
    name = "literature"
    stage = Stage.GATED

    def __init__(self, enabled: bool = False, provider: str = "auto",
                 model: Optional[str] = None, max_tokens: int = 1024, votes: int = 5,
                 cache_path: str = _DEFAULT_CACHE, names: Optional[dict] = None):
        self.names = names or {}                 # node id -> human-readable name (e.g. ENSG->gene symbol)
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.votes = votes                       # best-of-N majority vote (1 = single call)
        self.cache_path = cache_path
        self._reasoner = None
        self._resolved: Optional[str] = None
        self._initialized = False
        self._pcache: Optional[dict[str, dict]] = None   # feature-hash -> verdict record

    # --- persistent, feature-hashed cache (reused across runs) -----------
    def _feature_key(self, graph: TypedInteractionGraph, u: str, v: str) -> str:
        """Content hash of exactly what the judge sees for this pair — so an
        identical pair+context reuses the verdict, and a changed context re-judges."""
        a, b = sorted((u, v))
        feats = {"a": a, "b": b,
                 "a_type": graph.node_type.get(a), "b_type": graph.node_type.get(b),
                 "a_deg": graph.degree(a), "b_deg": graph.degree(b),
                 "model": self.model, "votes": self.votes}
        return hashlib.sha1(json.dumps(feats, sort_keys=True).encode()).hexdigest()

    def _load_cache(self) -> None:
        self._pcache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self._pcache[r["key"]] = r
                    except Exception:
                        continue

    def _store_cache(self, key: str, sc: StreamScore) -> None:
        rec = {"key": key, "value": sc.value, "flags": sc.flags, "evidence": sc.evidence}
        self._pcache[key] = rec
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a") as fh:              # append-only, O(1)
            fh.write(json.dumps(rec) + "\n")

    def _resolve_provider(self) -> Optional[str]:
        if self.provider == "auto":
            for p, env in _PROVIDER_ENV.items():
                if os.getenv(env):
                    return p
            return None
        return self.provider if os.getenv(_PROVIDER_ENV[self.provider]) else None

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            return
        prov = self._resolve_provider()
        if prov is None:
            return
        try:
            from ..llm import LLMConfig, LLMController, LiteratureReasoner
            self._reasoner = LiteratureReasoner(
                LLMController(LLMConfig(provider=prov, model=self.model)),
                max_tokens=self.max_tokens,
            )
            self._resolved = prov
        except Exception:
            self._reasoner = None

    def _skip(self, reason: str) -> StreamScore:
        return StreamScore(self.name, value=None,
                           evidence={"gated_status": "skipped", "reason": reason})

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        self._ensure()
        if self._pcache is None:
            self._load_cache()
        key = self._feature_key(graph, u, v)
        if key in self._pcache:                          # reuse a prior run's verdict
            r = self._pcache[key]
            return StreamScore(self.name, value=r["value"],
                               flags=list(r.get("flags") or []),
                               evidence=dict(r.get("evidence") or {}))
        if self._reasoner is None:                       # no key/disabled -> abstain (uncached)
            return self._skip("disabled" if not self.enabled else "no_api_key")

        ctx = {"u_type": graph.node_type.get(u), "v_type": graph.node_type.get(v),
               "u_degree": graph.degree(u), "v_degree": graph.degree(v)}
        if self.names.get(u):                    # gene symbol / name so the judge can reason
            ctx["protein_a_name"] = self.names[u]
        if self.names.get(v):
            ctx["protein_b_name"] = self.names[v]
        try:
            card = self._reasoner.reason_vote(u, v, ctx, votes=self.votes)
        except Exception:
            return self._skip("llm_error")               # transient — do not cache

        verdict, conf = card.verdict, float(card.confidence)
        flags: list[str] = []
        if verdict == "safe_negative":
            value: Optional[float] = round(conf, 4)
        elif verdict == "suspected_false_negative":
            value = round(1.0 - conf, 4)
            flags = ["suspected_false_negative"]
        else:  # uncertain
            value = None
        sc = StreamScore(
            self.name, value=value, flags=flags,
            evidence={"gated_status": "reviewed", "verdict": verdict,
                      "verdict_confidence": round(conf, 4), "rationale": card.rationale,
                      # reported confidence for entropy fusion = panel unanimity
                      # (a split best-of-N vote is a guess; a unanimous one commits)
                      "confidence": round(float(card.agreement), 4),
                      "evidence": card.evidence, "model": card.model,
                      "votes": card.n_votes, "agreement": card.agreement,
                      "vote_counts": card.vote_counts},
        )
        self._store_cache(key, sc)
        return sc


# Back-compat alias.
LiteratureStream = LiteratureFilter
