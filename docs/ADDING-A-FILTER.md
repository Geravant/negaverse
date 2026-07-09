# Adding a filter

A filter is a small independent module that scores a candidate pair. Adding one
touches **one file** and needs **no pipeline edits** — the registry discovers it.

## The 3 steps

1. **Subclass `Filter`**, declare its stage + modalities, implement `score()`.
2. **Register it** with `@register`.
3. **Import it** so registration runs (add to `negaverse/streams/__init__.py`).

```python
# negaverse/streams/colocalization.py
from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register


@register
class ColocalizationFilter(Filter):
    name = "colocalization"          # unique, stable; appears in provenance + config
    stage = Stage.GRADED             # VETO | GRADED | GATED
    modalities = frozenset({"ppi"})  # which interaction types this applies to

    def fit(self, graph: TypedInteractionGraph) -> None:
        # optional: precompute over the graph before scoring
        ...

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        # return one of:
        #   veto:    StreamScore(self.name, value=None, veto=True, evidence={...})
        #   abstain: StreamScore(self.name, value=None, evidence={"status": "no_data"})
        #   score:   StreamScore(self.name, value=0.0..1.0, flags=[...], evidence={...})
        return StreamScore(self.name, value=0.7, evidence={"reason": "different_compartment"})
```

That's it. The next `run_pipeline(..., PipelineConfig(modality="ppi"))` picks it up.

## The contract (`StreamScore`)

| field | meaning |
|---|---|
| `value` | contribution to *confidence that the pair is a true non-interaction*, in `[0,1]`; `None` = abstain |
| `veto` | `True` drops the candidate entirely (only meaningful for VETO-stage filters, but any stage may veto) |
| `flags` | strings attached to the emitted record (e.g. `"suspected_false_negative"`) |
| `evidence` | free-form dict kept in provenance; the embedding filter puts `topo` here (used for hardness) |

## Which stage?

- **VETO** — cheap, binary, "this pair is disqualified" (known positive, DB hit). Runs first; a veto drops the pair before any scoring.
- **GRADED** — cheap, graded, runs in parallel on survivors; `value` is merged into the confidence. Most biological filters live here (topology, chemistry, co-localization).
- **GATED** — expensive (LLM/literature). Runs only on the contested tail (near-boundary / low-confidence). Return `value=None` to abstain cheaply on the rest.

## Selecting & weighting filters

- Default: `PipelineConfig(modality="ppi")` runs every filter registered for that modality, in registration order.
- Subset: `PipelineConfig(filters=["known_positive_veto", "structured", "topology"])`.
- Weights: `PipelineConfig(weights={"structured": 1.0, "topology": 2.0})` — per-filter weight in the merge (default 1.0).

## Ablation

Drop a filter from `filters` (or set its weight to 0) and re-run the benchmark —
per-filter sub-scores and provenance are already in the output, so the effect is
directly measurable.
