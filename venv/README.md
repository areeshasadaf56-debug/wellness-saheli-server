# PCOS Prediction — Machine Learning Project

Predicting Polycystic Ovary Syndrome (PCOS) diagnosis using two separate
datasets, each treated as an independent experiment throughout.

## Datasets

1. **PCOS_infertility.csv** — 541 rows, 6 columns (Sl. No, Patient File No.,
   PCOS (Y/N), I beta-HCG, II beta-HCG, AMH). A small, focused dataset.
2. **PCOS_data_without_infertility.xlsx** (sheet `Full_new`) — 541 rows,
   44 columns covering demographics, vitals, hormones, symptoms, and
   ultrasound measurements. A much richer feature set.

These two datasets are **never merged** — every script processes and
reports on them completely separately, since they represent different
patient feature sets.

## Project Structure
## Setup

**1. Create and activate a virtual environment**

Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies (with the venv active)**
```powershell
python -m pip install -r requirements.txt
```

**3. Verify the install**
```powershell
python -c "import sklearn, pandas, numpy, openpyxl, matplotlib, seaborn, xgboost, imblearn; print('All imports OK')"
```

**4. Add your raw data**

Place the two original files into `data/raw/`, named exactly:
- `PCOS_infertility.csv`
- `PCOS_data_without_infertility.xlsx`

## How to Run (in order)

Each script depends on the output of the one before it — run them in this exact order:

```powershell
python src/data_cleaning.py
python src/preprocessing.py
python src/models.py
python src/feature_reduction.py
python src/evaluate_compare.py
```

| Script | What it does | Key outputs |
|---|---|---|
| `data_cleaning.py` | Cleans both datasets: fixes units, Yes/No columns, Blood Group, beta-HCG cross-fill, drops unrecoverable rows | `data/cleaned/*_clean.csv` |
| `preprocessing.py` | 80/20 stratified split, StandardScaler (fit on train only) | `data/cleaned/split_*/` |
| `models.py` | Trains 7 classifiers, computes Accuracy/Precision/Recall/F1, confusion matrices, 5-fold CV | `results/*/confusion_matrix_*.png`, `full_features_metrics.json` |
| `feature_reduction.py` | PCA (95% variance) and SelectKBest, re-runs all 7 models on each | `results/*/pca_metrics.json`, `selectkbest_metrics.json`, plots |
| `evaluate_compare.py` | Builds comparison table, grouped accuracy bar chart, writes report.md | `results/*/comparison_table.csv`, `accuracy_comparison.png`, `report.md` |

## Reproducibility

`random_state=42` is used consistently across the train/test split, all
tree-based/ensemble models, SVM, PCA, and RFE — so re-running the full
pipeline should produce the same results every time on the same machine.

## Summary of Findings

*(Fill this in from your actual `results/infertility_dataset/report.md`
and `results/without_infertility_dataset/report.md` once you've run the
full pipeline — paste your best model, accuracy, and feature-reduction
conclusion for each dataset here.)*

### Infertility Dataset
- Best model: _TBD_
- Feature reduction effect: _TBD_

### Without-Infertility Dataset
- Best model: _TBD_
- Feature reduction effect: _TBD_