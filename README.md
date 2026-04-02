# Wafer Defect Clustering & Yield Impact Analyzer

Spatial defect analysis pipeline for semiconductor wafer inspection data.

## Features
- DBSCAN clustering to classify defects as systematic (process-induced) vs. random (Poisson-distributed)
- Die yield modeling using Poisson, Murphy, and Seeds (negative binomial) approximations
- Defect density (D0) calculation with edge exclusion zone handling
- Wafer map visualization with cluster boundary rendering and yield annotation
- Sensitivity validation across simulated defect injection scenarios

## Usage
```bash
pip install -r requirements.txt
python wafer_analyzer.py
```

## Yield Models
| Model | Formula |
|---|---|
| Poisson | Y = exp(-D0 × A) |
| Murphy | Y = ((1 - exp(-D0×A)) / (D0×A))² |
| Seeds | Y = (1 + D0×A/α)^(-α) |
