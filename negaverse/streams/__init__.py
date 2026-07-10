"""Scoring filters + the hourglass staging registry.

Importing this package registers the built-in filters. Add your own by
subclassing Filter and decorating with @register (docs/ADDING-A-FILTER.md).
"""
from .base import Filter, Stream, Stage
from .registry import register, registered, build_filters
from .structured import KnownPositiveVeto, StructuredStream
from .topology import TopologyFilter
from .manifold import ManifoldSurprisalFilter, SequenceManifoldFilter
from .rules import RuleGradedFilter, RuleVetoFilter
from .literature import LiteratureFilter, LiteratureStream

__all__ = [
    "Filter", "Stream", "Stage",
    "register", "registered", "build_filters",
    "KnownPositiveVeto", "StructuredStream", "TopologyFilter",
    "ManifoldSurprisalFilter", "SequenceManifoldFilter",
    "RuleGradedFilter", "RuleVetoFilter",
    "LiteratureFilter", "LiteratureStream",
]
