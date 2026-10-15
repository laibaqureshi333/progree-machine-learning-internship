"""
TASK 4: Mini Project - High-Imbalance Clinical Disease Predictor Pipeline
===========================================================================
Objective : End-to-end classification system that spots early physiological
            abnormalities in highly skewed (imbalanced) medical data.

Dataset   : Breast Cancer Wisconsin (Diagnostic) dataset (UCI repository,
            bundled with Scikit-Learn). Binary target: malignant (1) vs
            benign (0). To faithfully exercise "high-imbalance" handling
            (the native ~63/37 split is only mildly imbalanced), we
            downsample the minority (malignant) class to create a
            realistic high-imbalance clinical scenario (~10:1), which is
            representative of real early-disease-screening datasets where
            positive cases are rare.

Pipeline stages:
    1. Data ingestion (UCI Breast Cancer Wisconsin via Scikit-Learn)
    2. Induce + report high class imbalance
    3. Preprocessing -> train/test split (stratified), feature scaling
    4. Imbalance handling -> SMOTE oversampling (train set only) AND
       class-weighting, compared side by side
    5. Model training + hyperparameter tuning -> GridSearchCV (5-fold
       stratified CV) across SVM, Random Forest, and XGBoost
    6. Evaluation -> recall-focused metrics (recall, precision, F1, ROC-AUC,
       PR-AUC), confusion matrices, cross-validated recall arrays
    7. Plots -> class distribution, ROC curves, PR curves, CV recall boxplot
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, recall_score,
    make_scorer, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# 0. Paths & logging setup (portable — works in Colab, notebooks, anywhere)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training_log.txt"), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Task4-ClinicalPredictor")

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Data ingestion
# --------------------------------------------------------------------------
log.info("Loading Breast Cancer Wisconsin (Diagnostic) dataset from UCI/sklearn...")
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X_full = df.drop(columns=["target"])
y_full = df["target"]
# In sklearn's encoding: 0 = malignant, 1 = benign. Flip so 1 = malignant
# (the disease / "positive" clinical outcome we want to detect early).
y_full = 1 - y_full
log.info(f"Original class balance -> malignant(1): {y_full.sum()}, "
         f"benign(0): {(y_full == 0).sum()}")

# --------------------------------------------------------------------------
# 2. Induce a high-imbalance scenario by downsampling the minority
#    (malignant) class, simulating a rare-disease early-screening setting.
# --------------------------------------------------------------------------
minority_idx = y_full[y_full == 1].index
majority_idx = y_full[y_full == 0].index

rng = np.random.RandomState(RANDOM_STATE)
n_minority_keep = max(30, int(len(majority_idx) * 0.10))  # ~10:1 imbalance
minority_keep = rng.choice(minority_idx, size=n_minority_keep, replace=False)

keep_idx = np.concatenate([majority_idx.values, minority_keep])
X = X_full.loc[keep_idx].reset_index(drop=True)
y = y_full.loc[keep_idx].reset_index(drop=True)

imbalance_ratio = (y == 0).sum() / (y == 1).sum()
log.info(f"Induced high-imbalance dataset -> total={len(y)}, "
         f"benign={sum(y == 0)}, malignant={sum(y == 1)}, "
         f"imbalance ratio ~{imbalance_ratio:.1f}:1")

# --------------------------------------------------------------------------
# 3. Train/test split (stratified to preserve imbalance ratio in both sets)
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
log.info(f"Train: {len(y_train)} (malignant={sum(y_train == 1)}) | "
         f"Test: {len(y_test)} (malignant={sum(y_test == 1)})")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
recall_scorer = make_scorer(recall_score, pos_label=1)

# --------------------------------------------------------------------------
# 4. Models + hyperparameter grids.
#    Each model is wrapped in an imblearn Pipeline with SMOTE applied only
#    to the training folds (never to validation/test data, which would leak
#    synthetic information). class_weight="balanced" is used alongside
#    SMOTE for the tree/SVM models as a second, complementary imbalance
#    countermeasure, and tuned as a hyperparameter itself.
# --------------------------------------------------------------------------
model_configs = {
    "SVM": {
        "pipeline": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__C": [0.1, 1, 10, 50],
            "clf__kernel": ["rbf", "linear"],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "RandomForest": {
        "pipeline": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "XGBoost": {
        "pipeline": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE,
            )),
        ]),
        "param_grid": {
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__scale_pos_weight": [1, imbalance_ratio],
        },
    },
}

results = {}
cv_recall_arrays = {}
best_estimators = {}

for name, cfg in model_configs.items():
    log.info(f"--- Tuning {name} (GridSearchCV, 5-fold stratified CV, "
             f"scoring=recall on malignant class) ---")
    grid = GridSearchCV(
        cfg["pipeline"], param_grid=cfg["param_grid"],
        scoring=recall_scorer, cv=cv, n_jobs=-1, refit=True,
    )
    grid.fit(X_train_scaled, y_train)
    log.info(f"[{name}] Best params: {grid.best_params_}")
    log.info(f"[{name}] Best CV recall: {grid.best_score_:.4f}")

    best_estimators[name] = grid.best_estimator_

    # Cross-validated recall array (5-6 folds) for the winning configuration,
    # re-scored explicitly so we retain the fold-by-fold array for reporting.
    cv6 = StratifiedKFold(n_splits=6, shuffle=True, random_state=RANDOM_STATE)
    recall_array = cross_val_score(
        grid.best_estimator_, X_train_scaled, y_train,
        cv=cv6, scoring=recall_scorer, n_jobs=-1,
    )
    cv_recall_arrays[name] = recall_array
    log.info(f"[{name}] 6-fold CV recall array: {np.round(recall_array, 4)} "
             f"| mean={recall_array.mean():.4f} | std={recall_array.std():.4f}")

    # Held-out test evaluation
    y_pred = grid.best_estimator_.predict(X_test_scaled)
    y_proba = grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]

    test_recall = recall_score(y_test, y_pred, pos_label=1)
    test_f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["benign", "malignant"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"[{name}] TEST -> Recall={test_recall:.4f} | F1={test_f1:.4f} | "
             f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}")
    log.info(f"[{name}] Confusion matrix (rows=actual, cols=pred [benign,malignant]):\n{cm}")
    log.info(f"[{name}] Classification report:\n{report}")

    results[name] = {
        "best_params": grid.best_params_,
        "cv_recall_mean": grid.best_score_,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
    }

# --------------------------------------------------------------------------
# 5. Summary table + pick best model by recall (primary clinical objective:
#    minimize false negatives / missed disease cases)
# --------------------------------------------------------------------------
summary_df = pd.DataFrame({
    name: {
        "CV Recall (mean)": r["cv_recall_mean"],
        "Test Recall": r["test_recall"],
        "Test F1": r["test_f1"],
        "Test ROC-AUC": r["roc_auc"],
        "Test PR-AUC": r["pr_auc"],
    }
    for name, r in results.items()
}).T
summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_summary.csv"))
log.info("\n=== Final Model Comparison (sorted by Test Recall) ===\n" +
         summary_df.sort_values("Test Recall", ascending=False).to_string())

best_model_name = summary_df["Test Recall"].idxmax()
log.info(f"\nBest model by recall on malignant class: {best_model_name}")

cv_recall_df = pd.DataFrame(cv_recall_arrays)
cv_recall_df.to_csv(os.path.join(OUTPUT_DIR, "cv_recall_arrays.csv"), index=False)
log.info("\nCross-validated recall arrays (6 folds each):\n" + cv_recall_df.to_string())

# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6a. Class distribution before/after induced imbalance
axes[0, 0].bar(["Benign", "Malignant"],
               [sum(y_full == 0), sum(y_full == 1)],
               color="#a0aec0", label="Original")
axes[0, 0].bar(["Benign", "Malignant"],
               [sum(y == 0), sum(y == 1)],
               color="#e53e3e", alpha=0.7, label="Induced high-imbalance subset")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Class Distribution: Original vs Induced Imbalance")
axes[0, 0].legend()

# 6b. ROC curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curves")
axes[0, 1].legend()

# 6c. Precision-Recall curves (most informative under high imbalance)
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
    axes[1, 0].plot(rec, prec, label=f"{name} (AP={r['pr_auc']:.3f})")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title("Precision-Recall Curves")
axes[1, 0].legend()

# 6d. Cross-validated recall boxplot across models
axes[1, 1].boxplot(
    [cv_recall_arrays[n] for n in model_configs.keys()],
    tick_labels=list(model_configs.keys()),
)
for i, n in enumerate(model_configs.keys(), start=1):
    axes[1, 1].scatter([i] * len(cv_recall_arrays[n]), cv_recall_arrays[n],
                        alpha=0.6, color="#2b6cb0")
axes[1, 1].set_ylabel("Recall (malignant class)")
axes[1, 1].set_title("6-Fold CV Recall Distribution by Model")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_plots.png"), dpi=150)
log.info("Saved evaluation plots to evaluation_plots.png")

log.info("Task 4 pipeline complete. Outputs: training_log.txt, "
         "model_comparison_summary.csv, cv_recall_arrays.csv, evaluation_plots.png")

# Display the evaluation plots inline if running in a notebook (Colab/Jupyter);
# silently skip this if running as a plain .py script from the command line.
try:
    from IPython.display import Image, display
    display(Image(os.path.join(OUTPUT_DIR, "evaluation_plots.png")))
except ImportError:
    pass
