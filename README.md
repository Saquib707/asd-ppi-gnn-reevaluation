# Hubs or Key Regulators? Reproduction code and data

Code and data for *Hubs or Key Regulators? Re-evaluating Graph Neural Networks on the
Autism Protein Interaction Network* (submitted to ICACECT 2027).

The pipeline builds an autism (ASD) protein–protein interaction network from SFARI Gene
and STRING v12.0 and labels genes by simulated Susceptible–Infected (SI) influence. It then
compares a graph attention network with a logistic-regression head (GAT-LR) with thirteen
alternatives, using identical features, labels and cross-validation folds. Further runs test
robustness, mechanism and leakage, and two control tasks on which degree is not the label.

## Pipeline

| Step | Script | Writes |
|---|---|---|
| 1 | `01_build_network.py` | `data/nodes.csv`, `data/edges.csv`, `data/network_stats.json` |
| 2 | `02_features_labels.py` | `data/features.csv`, `data/ic_scores.csv`, `data/label_sensitivity.json` |
| 3 | `03_models.py` | `results/metrics_raw.csv`, `metrics_summary.csv`, `significance.csv`, `rankings.csv`, `degree_overlap.json` |
| 4 | `04_ablation.py` | `results/exp_{horizon,threshold,residual,skip,inductive,importance}.csv` |
| 5 | `05_figures.py` | `../latex/figures/fig_ic_degree.pdf` (Fig. 1) |
| 6 | `06_sfari_check.py` | `results/exp_sfari.csv`, `results/sfari_diagnostics.json` |
| 7 | `07_tables.py` | `../latex/generated/numbers.tex` and table bodies |
| 9 | `09_tuning.py` | `results/exp_tuning.csv` (per-fold hyperparameter search) |
| 8 | `08_check.py` | pre-submission compliance report for the compiled PDF |

`gat_sparse.py` holds the edge-indexed GAT layer. It was checked against the dense
formulation in `03_models.py` and agrees to within 5 × 10⁻⁷.

## Running it

```bash
pip install -r requirements.txt        # Python 3.14.6 was used
python 01_build_network.py             # uses the cached inputs in data/
python 02_features_labels.py
python 03_models.py
python 04_ablation.py                  # --only / --exclude select experiments
python 05_figures.py
python 06_sfari_check.py
python 09_tuning.py                    # fair per-fold tuning of every model
python 07_tables.py
cd ../latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../experiments && python 08_check.py
```

`04_ablation.py`, `06_sfari_check.py` and `09_tuning.py` accept `--smoke`, which runs one fold per
experiment for 5 epochs as a quick test of the code path.

## Data provenance

- **SFARI Gene** export retrieved on 2026-09-10 (1,289 genes).
- **STRING v12.0** network for *Homo sapiens*, combined score ≥ 0.400. It is retrieved in
  a single API call: querying the gene list in chunks drops every interaction between chunks.
- SHA-256 checksums of both inputs are recorded in `data/network_stats.json`.

The exact input snapshots are included in `data/` (see `DATA.md`), so the results reproduce
precisely. Step 1 reuses them when present; `--refresh` downloads everything again. SFARI
Gene is updated over time, so a fresh export gives a somewhat different gene set. Please
cite SFARI Gene and STRING and follow their terms of use.

## Reproducibility

- Seeds: 42 for the SI simulation, 42 + r for the data splits of repeat r, and 42 + 10r + f
  for GNN initialisation in fold f.
- Hardware: Intel Core i5-8265U (4 cores / 8 threads), 16 GB RAM, CPU only.
- The GAT workload is overhead-bound on a graph this size (136 ms/epoch on one thread,
  111 ms on four). Steps 4 and 6 therefore run many single-threaded worker processes
  rather than using a GPU.

## Statistics

The paper compares models on matched folds with the corrected resampled t-test of Nadeau
and Bengio (2003). Folds of repeated cross-validation share training data, so tests that
treat them as independent overstate significance. All p-values in the paper come from
`07_tables.py`. The Wilcoxon p-values in `results/significance.csv` (step 3) and in the
console output and `sfari_diagnostics.json` of step 6 are uncorrected, are kept only for
quick inspection, and are not used in the paper.

## Deviation from the protocol being evaluated

The protocol being re-evaluated runs the SI process "until no new infections occur". On a
connected graph that infects every gene from every seed, so all influence scores become
equal. This code therefore uses a fixed horizon T, chosen by maximising the coefficient of
variation of the influence scores. Horizons 1 to 5 are reported as a sensitivity analysis.
