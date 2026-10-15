# Task 2 — Predictive Linear Regression Model Pipeline

## Objective
Predict a continuous numerical value (median house price) using a regularized
regression model.

## How to run
```bash
cd code
python task2_ridge_regression.py
```
Outputs are written to an auto-created `code/output/` folder (or wherever you
run it from). The script will use `data/housing.csv` if you copy it next to
the script, otherwise it auto-downloads the dataset.

## What it does
1. Loads the California Housing dataset (`data/housing.csv`)
2. Preprocesses: mean imputation + scaling (numeric columns), most-frequent
   imputation + one-hot encoding (categorical column `ocean_proximity`)
3. Trains a `Ridge` regression model, tuning `alpha` via 5-fold `GridSearchCV`
4. Trains a plain `LinearRegression` baseline for comparison
5. Logs R², RMSE, MAE for both models
6. Saves diagnostic plots (predicted vs actual, residuals, alpha-tuning curve)

## Results (see `outputs/`)
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Ridge (α=1.0) | 0.6257 | 70,038 | 50,709 |
| LinearRegression | 0.6257 | 70,031 | 50,702 |

- `outputs/diagnostics.png` — 3-panel diagnostic plot
- `outputs/metrics_summary.csv` — the table above
- `outputs/ridge_coefficients.csv` — feature importances
- `outputs/training_log.txt` — full run log
