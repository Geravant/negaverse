"""negaverse — matched, confidence-scored negative datasets from interaction graphs.

Walking-skeleton prototype (see docs/ARCHITECTURE.md §8.5).
"""
from .graph import TypedInteractionGraph
from .schema import NegativeRecord, StreamScore
from .pipeline import run_pipeline, PipelineConfig, PipelineResult

__all__ = [
    "TypedInteractionGraph",
    "NegativeRecord",
    "StreamScore",
    "run_pipeline",
    "PipelineConfig",
    "PipelineResult",
]
__version__ = "0.1.0"
