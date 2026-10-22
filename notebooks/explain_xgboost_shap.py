
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ml_dataset.csv"
MODEL_PATH = ROOT / "models" / "xgboost_weighted_demo.joblib"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load model bundle and dataset
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_columns"]

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=feature_cols).copy()

# Use a reproducible sample to keep SHAP computation manageable
sample_size = min(500, len(df))
sample = df.sample(n=sample_size, random_state=42)
X_sample = sample[feature_cols]

print("Records explained:", len(X_sample))
print("Features:", len(feature_cols))
print("Calculating SHAP values...")

# Explain the trained XGBoost model
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# Save global feature-importance plot
plt.figure()
shap.plots.bar(shap_values, max_display=11, show=False)
plt.title("SHAP Feature Importance — Weighted XGBoost")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Save summary plot showing direction and magnitude of contributions
shap.plots.beeswarm(shap_values, max_display=11, show=False)
plt.title("SHAP Summary — Weighted XGBoost")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_summary_plot.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Export mean absolute SHAP values as a table
importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_absolute_shap": abs(shap_values.values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    RESULTS_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\nTop features by mean absolute SHAP value:")
print(importance.head(11).to_string(index=False))

print("\nSaved:")
print("results/shap_feature_importance.png")
print("results/shap_summary_plot.png")
print("results/shap_feature_importance.csv")
