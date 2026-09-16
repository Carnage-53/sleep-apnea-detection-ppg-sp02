import pandas as pd
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = project_root / "data" / "raw" / "dataset_OSAS.pickle"

print("=" * 60)
print("OSASUD TIMESTAMP CONTINUITY CHECK")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_pickle(dataset_path)

print("Dataset loaded successfully!")

# ---------------------------------------------------------
# First 20 timestamps
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 20 TIMESTAMPS")
print("=" * 60)

print(df[["patient", "timestamp_datetime", "event", "anomaly"]].head(20).to_string(index=False))

# ---------------------------------------------------------
# Time differences
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TIMESTAMP DIFFERENCES")
print("=" * 60)

time_diff = df["timestamp_datetime"].diff()

print("\nFirst 20 time differences:")
print(time_diff.head(20))

print("\nMost common time differences:")
print(time_diff.value_counts().head(10))

# ---------------------------------------------------------
# PPG lengths
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PPG LENGTH CHECK")
print("=" * 60)

ppg_lengths = df["signal_pleth"].apply(len)

print("Unique PPG lengths:")
print(ppg_lengths.value_counts().head(10))

print("\nMinimum PPG length:", ppg_lengths.min())
print("Maximum PPG length:", ppg_lengths.max())

# ---------------------------------------------------------
# Check one patient
# ---------------------------------------------------------

patient_id = df["patient"].iloc[0]

patient_df = df[df["patient"] == patient_id].copy()

print("\n" + "=" * 60)
print(f"PATIENT {patient_id} TIME RANGE")
print("=" * 60)

print("Start:", patient_df["timestamp_datetime"].min())
print("End  :", patient_df["timestamp_datetime"].max())

print("Number of rows:", len(patient_df))

print("\n" + "=" * 60)
print("TIMESTAMP CHECK COMPLETE")
print("=" * 60)