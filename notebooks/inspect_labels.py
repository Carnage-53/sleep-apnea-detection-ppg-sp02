import pandas as pd
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = project_root / "data" / "raw" / "dataset_OSAS.pickle"

print("=" * 60)
print("OSASUD LABEL INSPECTION")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_pickle(dataset_path)

print("Dataset loaded successfully!")

# ---------------------------------------------------------
# EVENT LABELS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("UNIQUE EVENT LABELS")
print("=" * 60)

print(df["event"].unique())

print("\nEvent counts:")
print(df["event"].value_counts(dropna=False))

# ---------------------------------------------------------
# ANOMALY LABEL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ANOMALY LABELS")
print("=" * 60)

print(df["anomaly"].unique())

print("\nAnomaly counts:")
print(df["anomaly"].value_counts(dropna=False))

# ---------------------------------------------------------
# CROSS-TABULATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EVENT vs ANOMALY")
print("=" * 60)

print(pd.crosstab(df["event"], df["anomaly"]))

# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PATIENT INFORMATION")
print("=" * 60)

print("Number of patients:", df["patient"].nunique())

print("\nSamples per patient:")
print(df["patient"].value_counts().sort_index())

print("\n" + "=" * 60)
print("LABEL INSPECTION COMPLETE")
print("=" * 60)