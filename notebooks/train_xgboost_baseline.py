
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "ml_dataset.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing dataset: {INPUT}")

df = pd.read_csv(INPUT)

feature_columns = [
    "spo2",
    "ppg_mean",
    "ppg_std",
    "ppg_min",
    "ppg_max",
    "ppg_range",
    "ppg_median",
    "ppg_iqr",
    "ppg_rms",
    "ppg_mean_abs_diff",
    "ppg_slope",
]

X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
y = df["target"].astype(int)

# Remove invalid rows before splitting.
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]

if y.nunique() != 2:
    raise ValueError("Both target classes must be present.")

# This is a software-pipeline demonstration only:
# the current data contains one patient, so it cannot support
# patient-independent evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4,
)

print("Training XGBoost baseline...")
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\nTRAINING PIPELINE RESULTS")
print("Training records:", len(X_train))
print("Test records:", len(X_test))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    target_names=["NONE", "EVENT-LABELLED"],
    zero_division=0,
))

# Save model and predictions for reproducibility.
import joblib

joblib.dump(model, MODELS / "xgboost_baseline_demo.joblib")

pd.DataFrame({
    "actual_target": y_test.to_numpy(),
    "predicted_target": pred,
    "predicted_event_probability": prob,
}).to_csv(RESULTS / "xgboost_demo_predictions.csv", index=False)

cm = confusion_matrix(y_test, pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NONE", "EVENT-LABELLED"],
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("XGBoost Baseline — Demonstration Split")
fig.tight_layout()
fig.savefig(RESULTS / "xgboost_demo_confusion_matrix.png", dpi=200)
plt.close(fig)

print("\nModel saved:", MODELS / "xgboost_baseline_demo.joblib")
print("Confusion matrix saved:", RESULTS / "xgboost_demo_confusion_matrix.png")
print("\nCAUTION: This is a single-patient demonstration split,")
print("not a valid estimate of generalization to unseen patients.")
