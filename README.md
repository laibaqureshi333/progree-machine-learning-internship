# Progree — Machine Learning Tasks Submission

This folder contains the complete solutions for Tasks 1–4.

```
Submission/
├── README.md                              <- you are here
├── Task1_LinkedIn_Announcement/
│   ├── INSTRUCTIONS.md                    <- completed, with final caption
│   └── linkedin_post_screenshot.png       <- proof of completion
├── Task2_Regression_Pipeline/
│   ├── README.md
│   ├── code/task2_ridge_regression.py
│   ├── data/housing.csv
│   └── outputs/  (plots, metrics, logs)
├── Task3_CNN_Image_Classification/
│   ├── README.md
│   ├── code/task3_cnn_classifier.py
│   └── outputs/  (plots, classification report, logs)
└── Task4_Clinical_Imbalance_Predictor/
    ├── README.md
    ├── code/task4_clinical_predictor.py
    └── outputs/  (plots, metrics, logs)
```

Each task folder is self-contained with its own README explaining the
objective, how to run it, and the results obtained. Every script has already
been run end-to-end; the `outputs/` folders contain the real logs, metrics,
and plots from that run — nothing here is hypothetical.

## Quick summary of results

| Task | Metric | Result |
|---|---|---|
| 1 — LinkedIn Announcement | Status | ✅ Posted (screenshot included) |
| 2 — Regression | R² / RMSE | 0.626 / $70,038 |
| 3 — CNN Classification | Test Accuracy | 95.6% |
| 4 — Clinical Predictor | Best Recall (SVM) | 100% |

**All four tasks are complete.**

## Environment setup
```bash
pip install scikit-learn pandas numpy matplotlib tensorflow imbalanced-learn xgboost
```

## Running in Google Colab
Upload any `code/*.py` file, then run:
```python
!python task2_ridge_regression.py   # (or task3_/task4_ equivalent)
```
Each script auto-creates its own `output/` folder and downloads any dataset
it needs — no manual path setup required.
