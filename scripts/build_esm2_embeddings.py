"""Compute ESM2 protein embeddings from sequences -> a .npz for SequenceManifoldFilter.

Input: a TSV of `id <tab> sequence` (one protein per line).
Output: `<out>.npz` with `ids` and `emb` (mean-pooled last-hidden-state per protein),
the format `SequenceManifoldFilter(path=...)` / `load_embeddings_npz` expect.

    PYTHONPATH=. python3 scripts/build_esm2_embeddings.py \
        --seqs local-docs/dryad-ppi/sequences.tsv --out out/esm2.npz

Model: facebook/esm2_t6_8M_UR50D by default (small, CPU-friendly, 320-d). Use a
larger checkpoint (…t33_650M…, 1280-d) to raise the sequence axis's ceiling.

Note on HuRI: its nodes are Ensembl gene ids, so key the emitted `ids` by the
graph's node ids (map Ensembl→UniProt→sequence first). The filter matches ids
directly and abstains for anything unmatched — so partial coverage is safe.
"""
from __future__ import annotations

import argparse

import numpy as np


def _load_seqs(path: str, max_len: int) -> tuple[list[str], list[str]]:
    ids, seqs = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            ids.append(parts[0])
            seqs.append(parts[1][:max_len])
    return ids, seqs


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_esm2_embeddings")
    ap.add_argument("--seqs", required=True, help="TSV: id <tab> sequence")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--limit", type=int, default=0, help="cap #proteins (0 = all)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    ids, seqs = _load_seqs(args.seqs, args.max_len)
    if args.limit:
        ids, seqs = ids[:args.limit], seqs[:args.limit]
    print(f"loaded {len(ids)} sequences; loading {args.model} ...")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len).to(device)
            out = model(**enc).last_hidden_state          # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1)     # (B, L, 1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean over residues
            vecs.append(pooled.cpu().numpy().astype("float32"))
            print(f"  {min(i + args.batch, len(seqs))}/{len(seqs)}", end="\r")

    emb = np.concatenate(vecs, axis=0)
    np.savez_compressed(args.out, ids=np.array(ids), emb=emb)
    print(f"\nwrote {args.out}  ({emb.shape[0]} x {emb.shape[1]})")


if __name__ == "__main__":
    main()
