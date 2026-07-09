"""Downstream-model benchmark — the hypothesis test (docs/IMPLEMENTATION-PLAN §Phase 1).

Do negaverse's hard negatives train a better link-prediction model than random
negatives? Trains on positives + (random | negaverse) negatives and evaluates on
a held-out, unbiased test set (AUROC / AUPRC).
"""
from .benchmark import run_benchmark, BenchmarkResult

__all__ = ["run_benchmark", "BenchmarkResult"]
