
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "ppg_features.csv"
OUTPUT = ROOT / "data" / "processed" / "ml_dataset.csv"

print("ML DATASET PREPARATION")
print("-" * 40)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Feature table not found: {INPUT}"
    )

df = pd.read_csv(INPUT)
print("Input records:", len(df))

# Binary target: NONE = 0, annotated respiratory event = 1.
df["target"] = (df["event"] != "NONE").astype(int)

# Only use numerical measurements and PPG-derived features as predictors.
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

# Keep the target and patient identifier separate from predictors.
required_columns = feature_columns + ["target", "patient"]
dataset = df[required_columns].copy()

# Remove rows with missing predictors or non-finite values.
dataset[feature_columns] = dataset[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)

dataset = dataset.dropna(subset=feature_columns + ["target", "patient"])
dataset = dataset[
    dataset[feature_columns].apply(
        lambda col: col.map(lambda x: abs(x) != float("inf"))
    ).all(axis=1)
].copy()

dataset["target"] = dataset["target"].astype(int)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT, index=False)

print("\nCLASS COUNTS")
print(dataset["target"].value_counts().sort_index().rename(
    index={0: "NONE (0)", 1: "EVENT-LABELLED (1)"}
).to_string())

print("\nPredictor columns:", len(feature_columns))
print("Usable records:", len(dataset))
print("Patients:", dataset["patient"].nunique())
print("Saved:", OUTPUT)

print("\nIMPORTANT:")
print("- Target 1 means an annotated respiratory event, not a diagnosis.")
print("- This sample is from one patient and is not suitable for")
print("  patient-independent model evaluation.")
