# Data provenance

This repository includes the exact input files used for the paper, so the results can be
reproduced rather than approximated. Both sources are third-party databases; please cite
them and follow their own terms of use.

## Files

| File | Source | Retrieved | SHA-256 (first 16) |
|---|---|---|---|
| `data/sfari_genes_raw.csv` | SFARI Gene export (1,289 genes, with evidence score, syndromic flag, EAGLE score, report count) | 2026-09-10 | `ec3b830ab3f4e185` |
| `data/string_network_raw.tsv` | STRING v12.0 API, `network` endpoint, *Homo sapiens*, combined score ≥ 0.400, one query for the whole gene list | 2026-09-10 | `9f30fd0cc96f1c21` |
| `data/string_id_map.tsv` | STRING v12.0 API, `get_string_ids` endpoint (gene symbol → preferred name) | 2026-09-10 | — |

The full checksums are in `data/network_stats.json`. `01_build_network.py` uses these cached
files when present and re-downloads them with `--refresh`.

Everything else under `data/` is derived by this code: `nodes.csv`, `edges.csv`,
`features.csv`, `ic_scores.csv`, `label_sensitivity.json`, `network_stats.json`.

## Why the snapshot is included

SFARI Gene is updated continuously, so a fresh export gives a different gene set and
slightly different numbers. The frozen snapshot is what the paper's results were computed
from.

## Please cite the sources

- SFARI Gene: B. S. Abrahams, D. E. Arking, D. B. Campbell *et al.*, "SFARI Gene 2.0: a
  community-driven knowledgebase for the autism spectrum disorders (ASDs)," *Molecular
  Autism*, vol. 4, art. 36, 2013.
- STRING: D. Szklarczyk, R. Kirsch, M. Koutrouli *et al.*, "The STRING database in 2023:
  protein–protein association networks and functional enrichment analyses for any sequenced
  genome of interest," *Nucleic Acids Research*, vol. 51, no. D1, pp. D638–D646, 2023.

Consult https://gene.sfari.org and https://string-db.org for the current terms that apply to
their data. If either changes in a way that makes redistribution inappropriate, delete the
file here; `01_build_network.py` downloads it again.
