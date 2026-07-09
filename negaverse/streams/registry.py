"""Filter registry — the frictionless add/modify surface.

Adding a filter is: subclass `Filter`, decorate the class with `@register`, and
(optionally) name it in the pipeline config. The pipeline discovers active
filters by modality; no orchestration code changes. See docs/ADDING-A-FILTER.md.
"""
from __future__ import annotations

from .base import Filter

_REGISTRY: dict[str, type[Filter]] = {}


def register(cls: type[Filter]) -> type[Filter]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must set a `name` before @register")
    _REGISTRY[cls.name] = cls
    return cls


def registered() -> dict[str, type[Filter]]:
    return dict(_REGISTRY)


def build_filters(modality: str = "ppi", names: list[str] | None = None) -> list[Filter]:
    """Instantiate the active filters for a modality (registration order).
    `names` overrides the default selection for that modality."""
    if names is None:
        names = [n for n, c in _REGISTRY.items() if modality in c.modalities]
    missing = [n for n in names if n not in _REGISTRY]
    if missing:
        raise KeyError(f"unknown filter(s): {missing}; registered: {list(_REGISTRY)}")
    return [_REGISTRY[n]() for n in names]
