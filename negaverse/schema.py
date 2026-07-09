"""Output contract for a negaverse run (ARCHITECTURE.md §7).

A negative is a proposed non-edge (u, v) that carries everything needed to
audit and re-score it later: a calibrated confidence, a hardness percentile,
the per-stream sub-scores that produced it, and a provenance trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Mode = Literal["train", "eval"]


@dataclass
class StreamScore:
    """One scoring stream's view of a candidate (ARCHITECTURE.md §5).

    value is the stream's contribution to *confidence that (u, v) is a true
    non-interaction*, in [0, 1]; None means the stream abstains (e.g. the
    literature stub, or no evidence found). veto=True is a hard override: the
    pair is a known/near-certain positive and must never be emitted.
    """

    stream: str
    value: Optional[float]
    veto: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def abstains(self) -> bool:
        return self.value is None and not self.veto


@dataclass
class NegativeRecord:
    u: str
    v: str
    mode: Mode
    confidence: float                     # calibrated [0,1] that the pair is a true non-interaction
    hardness: float                       # distance-to-positive-manifold percentile [0,1]
    streams: dict[str, Optional[float]]   # per-stream sub-scores: structured / literature / topology
    provenance: dict[str, Any]            # filters fired, evidence, source graph, versions
    flags: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten streams / flags for tabular output
        for name, val in self.streams.items():
            d[f"stream_{name}"] = val
        d.pop("streams")
        d["flags"] = ";".join(self.flags)
        d["provenance"] = self.provenance
        return d
