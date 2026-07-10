# Adding an external positive-interaction source

`rules/sources.yaml` is a manifest of extra "known positive" pair files fed to
the veto filter (`negaverse/streams/structured.py::KnownPositiveVeto`).

This is **not** the rule engine (`ppi.yaml`/`pli.yaml`, `AUTHORING.md`). A rule
scores how biologically plausible a non-edge is (`when` + `effect` + `weight`).
A source here scores nothing — it's a plain lookup table of pairs documented
as interacting *somewhere*, even if that pair isn't an edge in whatever graph
the pipeline happens to be loading for a given run. If a candidate matches any
pair in any listed source, it's vetoed — dropped, never emitted — exactly like
a positive edge already in the graph.

Why this exists: the graph loaded for a run (e.g. Gordon et al.'s 332
SARS-CoV-2 edges) is one experiment's worth of evidence, not the full universe
of known interactions. A candidate can be a true positive documented in
BioGRID/IntAct/STRING/etc. without appearing in that one graph. Listing those
databases here closes that gap — union-of-sources exclusion.

## Field contract

| field | required | meaning |
|---|---|---|
| `name` | ✓ | short slug, used in logs/provenance |
| `modality` | ✓ | `ppi` or `pli` — which engine side these pairs veto against (mirrors the `modality` field in `rules/ppi.yaml`/`rules/pli.yaml`) |
| `path` | ✓ | path to a local file of ID pairs (under `local-docs/`, gitignored) |
| `id_space` | ✓ | ID system the pairs use. For `ppi` sources: a single value (`uniprot`, `ensembl`, `gene_symbol`, ...) shared by both columns. For `pli` sources: `<protein_id_space>/<ligand_id_space>` (e.g. `uniprot/inchikey`), since column 1 is a protein ID and column 2 is a ligand ID in a different space. Must match the graph's node ID space or the pairs simply never match anything. |
| `description` | – | one line: what this source is |
| `source` | – | citation — name, version/date, URL |

## File format

Each `path` is a plain 2-column file (tab- or whitespace-separated), one pair
per line, `#`-comments allowed — the same shape as Negatome's files.

`ppi` sources (both columns share `id_space`):

```
P12345	Q9Y4K3
O00203	Q6ZNK6
```

`pli` sources (column 1 is protein, column 2 is ligand — a different ID space,
e.g. `id_space: uniprot/inchikey`):

```
P12345	BSYNRYMUTXBXSQ-UHFFFAOYSA-N
O00203	RYYVLZVUVIJVGH-UHFFFAOYSA-N
```

## Adding a source

1. Get the export (e.g. a BioGRID "All Interactions" TSV reduced to two ID
   columns) and place it under `local-docs/` — never commit it.
2. Add an entry to `sources.yaml`:
   ```yaml
   - name: biogrid_human
     modality: ppi
     path: local-docs/biogrid/biogrid_human_pairs.tsv
     id_space: uniprot
     description: BioGRID human-human physical interactions
     source: "BioGRID 4.4, https://thebiogrid.org, downloaded 2026-07-09"
   ```
3. Check `id_space` matches whatever graph you're running against. A
   UniProt-keyed source silently matches nothing against an Ensembl-keyed
   graph (HuRI) — no error, just zero effect. There's no automatic ID-space
   translation here (contrast `AUTHORING.md`'s annotation fields); if you need
   one, follow the pattern in `scripts/build_uniprot_ensembl_map.py`.
4. Re-run the pipeline — `negaverse/io/sources.py::load_positive_sources()`
   loads the manifest automatically and unions every listed file's pairs into
   `KnownPositiveVeto`. A missing file warns, it doesn't crash — a staged
   entry (declared before you've placed the file) is fine.

## Status

**Wired.** `negaverse/io/sources.py::load_positive_sources()` reads this manifest,
and `KnownPositiveVeto.fit()` (`negaverse/streams/structured.py`) loads every source
whose `path` exists — restricted to the current graph's node ids, so a PLI source
never matches a PPI graph — and vetoes those pairs (union-of-sources exclusion).
A run reports which sources loaded vs. were missing under
`stats["known_positive_sources"]`. The staged entries below all report `missing`
until a real 2-column file is placed at each `path`; then it activates with no code
change. (`KnownPositiveVeto` still also accepts a `known_positives` set directly.)
