
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ml_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load prepared data
df = pd.read_csv(DATA_PATH)

target_col = "target"
excluded_cols = ["target", "patient", "timestamp_datetime", "event"]

feature_cols = [
    col for col in df.columns
    if col not in excluded_cols
    and pd.api.types.is_numeric_dtype(df[col])
]

df = df.dropna(subset=feature_cols + [target_col]).copy()

X = df[feature_cols]
y = df[target_col].astype(int)

print("Records:", len(df))
print("Patients:", df["patient"].nunique())
print("\nClass counts:")
print(y.value_counts().sort_index().rename({
    0: "NONE",
    1: "EVENT-LABELLED"
}))

# Stratified train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Give the minority class more weight during training
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))
print("scale_pos_weight:", round(scale_pos_weight, 3))

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Select a threshold using validation data only
val_prob = model.predict_proba(X_val)[:, 1]

threshold_results = []
for threshold in np.arange(0.10, 0.91, 0.05):
    val_pred = (val_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, val_pred, average="binary", zero_division=0
    )
    threshold_results.append((threshold, precision, recall, f1))

threshold_table = pd.DataFrame(
    threshold_results,
    columns=["threshold", "precision", "recall", "f1"]
)

# Choose the threshold with the best validation F1 score
best_row = threshold_table.loc[threshold_table["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

threshold_table.to_csv(
    RESULTS_DIR / "xgboost_threshold_comparison.csv", index=False
)

print("\nValidation threshold comparison:")
print(threshold_table.round(3).to_string(index=False))
print("\nSelected threshold:", round(best_threshold, 2))
print("Validation F1:", round(float(best_row["f1"]), 3))

# Evaluate only once on the held-out test set
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nHELD-OUT TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, test_pred), 4
))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    labels=[0, 1],
    target_names=["NONE", "EVENT-LABELLED"],
    zero_division=0,
))

# Save test predictions and confusion matrix
predictions = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": test_pred,
    "event_probability": test_prob,
    "threshold": best_threshold,
})
predictions.to_csv(
    RESULTS_DIR / "xgboost_weighted_test_predictions.csv", index=False
)

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NONE", "EVENT-LABELLED"]
)
disp.plot(values_format="d")
plt.title("Weighted XGBoost — Held-Out Test Set")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "xgboost_weighted_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Save model and its selected threshold
joblib.dump(
    {
        "model": model,
        "feature_columns": feature_cols,
        "threshold": best_threshold,
    },
    MODEL_DIR / "xgboost_weighted_demo.joblib",
)

print("\nSaved model, threshold table, predictions and confusion matrix.")
print("Remember: this is a single-patient demonstration, not clinical validation.")
