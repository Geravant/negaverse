"""Load precomputed node embeddings (e.g. ESM2 protein embeddings).

Format: a `.npz` with an `ids` array (node identifiers) and an `emb` array
(one row per id). Produced by scripts/build_esm2_embeddings.py, and shipped with
some datasets (e.g. local-docs/dryad-ppi/esm2_t6_emb.npz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_embeddings_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Return {node_id: vector}. Keys are stringified so they match graph node ids."""
    data = np.load(path)
    ids, emb = data["ids"], data["emb"]
    return {str(i): emb[j].astype(float) for j, i in enumerate(ids)}
