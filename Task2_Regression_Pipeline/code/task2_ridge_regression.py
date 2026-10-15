"""
TASK 2: Predictive Linear Regression Model Pipeline
=====================================================
Objective : Build a supervised regression model that predicts a continuous
            numerical target (median house value) using the California
            Housing dataset.

Dataset   : California Housing (the classic 1990 census version, the same
            one distributed with Scikit-Learn / Aurelien Geron's book).
            It ships with a genuine categorical column (ocean_proximity)
            and genuine missing values (total_bedrooms), so every required
            preprocessing step below operates on real data imperfections
            rather than synthetic ones.

Pipeline stages:
    1. Data ingestion
    2. Preprocessing  -> mean imputation, feature scaling, categorical encoding
    3. Model training -> Ridge Regression (L2-regularized linear model),
                          with a plain Linear Regression baseline for comparison
    4. Hyperparameter tuning -> GridSearchCV over the Ridge alpha parameter
    5. Evaluation -> R^2 and RMSE, logged to console + saved to disk
    6. Diagnostic plots -> predicted vs actual, residuals, coefficients
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --------------------------------------------------------------------------
# 0. Paths & logging setup
#    Everything is written to an "output/" folder created next to this
#    script (or in the current working directory if run from a notebook),
#    so this runs unmodified on any machine — no hardcoded absolute paths.
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
log = logging.getLogger("Task2-RidgeRegression")

RANDOM_STATE = 42
DATA_PATH = os.path.join(BASE_DIR, "housing.csv")  # California Housing (1990 census)
DATA_URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/master/"
            "datasets/housing/housing.csv")

# --------------------------------------------------------------------------
# 1. Data ingestion
#    Uses the local housing.csv if it's sitting next to this script;
#    otherwise downloads it automatically from a public mirror.
# --------------------------------------------------------------------------
log.info("Loading California Housing dataset...")
if not os.path.exists(DATA_PATH):
    log.info(f"'{DATA_PATH}' not found locally — downloading from {DATA_URL} ...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)  # cache locally for next run
else:
    df = pd.read_csv(DATA_PATH)
log.info(f"Dataset shape: {df.shape}")
log.info(f"Columns: {list(df.columns)}")
log.info(f"Missing values per column:\n{df.isna().sum().to_string()}")

TARGET = "median_house_value"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = [c for c in X.columns if c != "ocean_proximity"]
categorical_features = ["ocean_proximity"]
log.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
log.info(f"Categorical features: {categorical_features} "
         f"-> classes: {sorted(X['ocean_proximity'].dropna().unique())}")

# --------------------------------------------------------------------------
# 2. Train / test split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
log.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# --------------------------------------------------------------------------
# 3. Preprocessing pipeline
#    - numeric: mean imputation -> standard scaling
#    - categorical: most-frequent imputation -> one-hot encoding
# --------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# --------------------------------------------------------------------------
# 4. Full modeling pipeline + hyperparameter search
# --------------------------------------------------------------------------
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(random_state=RANDOM_STATE)),
])

param_grid = {"regressor__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log.info("Running GridSearchCV over Ridge alpha values with 5-fold CV...")
grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_["regressor__alpha"]
log.info(f"Best alpha found via CV: {best_alpha}")
log.info(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# --------------------------------------------------------------------------
# 5. Baseline (plain Linear Regression) for comparison
# --------------------------------------------------------------------------
baseline_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
baseline_pipeline.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 6. Evaluation on held-out test set
# --------------------------------------------------------------------------
def evaluate(model, X_te, y_te, name):
    preds = model.predict(X_te)
    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae = mean_absolute_error(y_te, preds)
    log.info(f"[{name}] R^2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")
    return preds, r2, rmse, mae

log.info("Evaluating models on the held-out test set...")
ridge_preds, ridge_r2, ridge_rmse, ridge_mae = evaluate(
    best_model, X_test, y_test, f"Ridge (alpha={best_alpha})")
base_preds, base_r2, base_rmse, base_mae = evaluate(
    baseline_pipeline, X_test, y_test, "LinearRegression (baseline)")

metrics_summary = pd.DataFrame({
    "Model": [f"Ridge (alpha={best_alpha})", "LinearRegression"],
    "R2": [ridge_r2, base_r2],
    "RMSE": [ridge_rmse, base_rmse],
    "MAE": [ridge_mae, base_mae],
})
metrics_summary.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
log.info("\n" + metrics_summary.to_string(index=False))

# --------------------------------------------------------------------------
# 7. Diagnostic plots
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(y_test, ridge_preds, alpha=0.3, s=10, color="#2b6cb0")
lims = [min(y_test.min(), ridge_preds.min()), max(y_test.max(), ridge_preds.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5)
axes[0].set_xlabel("Actual Median House Value")
axes[0].set_ylabel("Predicted Median House Value")
axes[0].set_title(f"Ridge: Predicted vs Actual (R2={ridge_r2:.3f})")

residuals = y_test.values - ridge_preds
axes[1].scatter(ridge_preds, residuals, alpha=0.3, s=10, color="#38a169")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted Value")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

cv_results = pd.DataFrame(grid_search.cv_results_)
axes[2].plot(
    cv_results["param_regressor__alpha"].astype(float),
    -cv_results["mean_test_score"],
    marker="o", color="#805ad5",
)
axes[2].set_xscale("log")
axes[2].set_xlabel("Alpha (log scale)")
axes[2].set_ylabel("Mean CV RMSE")
axes[2].set_title("Regularization Strength vs CV RMSE")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "diagnostics.png"), dpi=150)
log.info("Saved diagnostic plots to diagnostics.png")

# --------------------------------------------------------------------------
# 8. Feature importance (Ridge coefficients, on scaled/encoded features)
# --------------------------------------------------------------------------
feature_names = numeric_features + list(
    best_model.named_steps["preprocessor"].named_transformers_["cat"]
    .named_steps["onehot"].get_feature_names_out(categorical_features)
)
coefs = best_model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs}).sort_values(
    "coefficient", key=np.abs, ascending=False
)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "ridge_coefficients.csv"), index=False)
log.info("\nTop feature coefficients (Ridge):\n" + coef_df.to_string(index=False))

log.info("Task 2 pipeline complete. Outputs: training_log.txt, "
         "metrics_summary.csv, ridge_coefficients.csv, diagnostics.png")
