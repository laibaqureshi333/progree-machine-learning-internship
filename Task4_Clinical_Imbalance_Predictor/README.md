# Task 4 — High-Imbalance Clinical Disease Predictor

## Objective
Build an end-to-end classifier that detects a rare medical condition
(malignant tumors) in a highly imbalanced dataset, optimizing for recall
(minimizing missed-disease false negatives).

## How to run
```bash
pip install scikit-learn imbalanced-learn xgboost matplotlib
cd code
python task4_clinical_predictor.py
```
Outputs are written to an auto-created `code/output/` folder.

## What it does
1. Loads the Breast Cancer Wisconsin (Diagnostic) dataset
2. Downsamples the malignant class to create a realistic ~10:1 imbalance
3. Splits data (stratified), scales features
4. Applies **SMOTE** oversampling inside an `imblearn.Pipeline` — applied
   only to training folds, never leaking into validation/test
5. Tunes **SVM**, **Random Forest**, and **XGBoost** via `GridSearchCV`
   (5-fold stratified CV, scored on recall)
6. Generates a 6-fold cross-validated recall array per model
7. Evaluates on the test set: recall, F1, ROC-AUC, PR-AUC, confusion matrix
8. Plots: class distribution, ROC curves, PR curves, CV-recall boxplot

## Results (see `outputs/`)
| Model | CV Recall | Test Recall | Test F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **SVM** | 0.927 | **1.000** | 1.000 | 1.000 | 1.000 |
| RandomForest | 0.840 | 0.889 | 0.941 | 1.000 | 1.000 |
| XGBoost | 0.920 | 0.889 | 0.941 | 0.998 | 0.980 |

SVM caught every malignant case in the test set (0 false negatives).

- `outputs/evaluation_plots.png` — 4-panel evaluation figure
- `outputs/model_comparison_summary.csv` — the table above
- `outputs/cv_recall_arrays.csv` — raw 6-fold recall arrays per model
- `outputs/training_log.txt` — full run log
