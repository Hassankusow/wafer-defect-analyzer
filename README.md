# Wafer Defect Clustering & Yield Impact Analyzer

[![Tests](https://github.com/Hassankusow/wafer-defect-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/Hassankusow/wafer-defect-analyzer/actions/workflows/tests.yml)

A spatial defect analysis pipeline for semiconductor wafer inspection data. Uses DBSCAN clustering to classify defects as systematic (process-induced) vs. random (Poisson-distributed), and models die yield using industry-standard approximations.

---

## Features

- **DBSCAN Clustering** — classifies defects as systematic vs. Poisson-distributed random
- **Three Yield Models** — Poisson, Murphy, and Seeds (negative binomial) die yield predictions
- **Defect Density (D0)** — per-wafer defect density calculation with edge exclusion zone handling
- **Wafer Map Visualization** — dual-panel maps: classification view and cluster ID view
- **Yield Impact Breakdown** — compares yield with and without systematic defect contribution
- **Configurable Geometry** — adjustable wafer radius, edge exclusion, and die size

---

## Tech Stack

Python, NumPy, Pandas, Scikit-learn (DBSCAN), Matplotlib, SciPy

---

## Usage

```bash
pip install -r requirements.txt
python wafer_analyzer.py
```

---

## Example Output

```
total_defects: 160
systematic_defects: 93
random_defects: 67
n_clusters: 6
defect_density_D0: 0.002357 def/mm²
active_area_mm2: 67886.68
die_size_mm2: 100

yield_poisson:           79.0%
yield_murphy:            79.37%
yield_seeds:             79.7%
yield_murphy_random_only: 90.68%
```

The gap between `yield_murphy` (79.37%) and `yield_murphy_random_only` (90.68%) quantifies the yield loss attributable specifically to systematic defect clusters — the key metric for process engineers targeting root cause elimination.

---

## Yield Models

| Model | Formula | Use Case |
|-------|---------|----------|
| Poisson | Y = exp(-D₀ × A) | Simple baseline |
| Murphy | Y = ((1 - exp(-D₀A)) / (D₀A))² | Industry standard for random defects |
| Seeds | Y = (1 + D₀A/α)^(-α) | Clustered defect distributions |

---

## API Reference

| Function | Description |
|----------|-------------|
| `simulate_defects(config, n_random, clusters)` | Generate mock wafer defect data with random + systematic components |
| `classify_defects(df, eps, min_samples)` | Run DBSCAN, label each defect as systematic or random |
| `murphy_yield(D0, A)` | Murphy yield model calculation |
| `poisson_yield(D0, A)` | Poisson yield model calculation |
| `seeds_yield(D0, A, alpha)` | Seeds (negative binomial) yield model |
| `compute_yield_analysis(df, config)` | Full yield report across all three models |
| `plot_wafer_map(df, analysis, config)` | Render dual-panel wafer defect map |

---

## Author

**Hassan Abdi**
[GitHub](https://github.com/Hassankusow) | [LinkedIn](https://linkedin.com/in/hassan-abdi-119357267)
