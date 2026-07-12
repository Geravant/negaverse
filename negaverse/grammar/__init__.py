"""A probabilistic language of biological evidence for negative-label safety.

The shipped rule stream (`streams/rules.py`) reduces every fired rule to one
weighted mean, discarding which evidence combinations occurred, coverage /
missingness, contradictions, and proof structure — and, fatally for safety, it
averages SAFE and RISK evidence into one semantic blob. This package keeps them
separate.

Staged, evaluate-first (docs/IG-FEATURES.md, the grammar arc):
  * features.py — a SPARSE proof fingerprint: branch-separated production counts
    (safe / risk / coverage / contradiction) from the streams already firing. No
    new knowledge graph; the cheap first rung.
  * (later, only if the fingerprint earns it) typed Horn/Datalog proofs, a
    class-conditional grammar likelihood, and Fisher-whitened proof embeddings.

Fisher geometry enters here as an evidence *representation* (whitened deviation
of a proof pattern), never as a selector — the D-optimal experiment
(`ig/fisher.py`, INFOGEO §3) showed Fisher magnitude selects hidden positives.
A proof's Fisher norm may flag an unusual pattern; it must never authorise a
label.
"""
from __future__ import annotations

from .features import fingerprint_matrix, fingerprint_names, proof_fingerprint

__all__ = ["proof_fingerprint", "fingerprint_matrix", "fingerprint_names"]
