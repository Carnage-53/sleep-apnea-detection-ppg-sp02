import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "raw" / "dataset_OSAS.pickle"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading OSASUD dataset...")
df = pd.read_pickle(dataset_path)

print("Dataset loaded successfully!")

# ---------------------------------------------------------
# DATASET SIZE
# ---------------------------------------------------------

total_rows = len(df)
total_patients = df["patient"].nunique()

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Total records:", total_rows)
print("Number of patients:", total_patients)

# ---------------------------------------------------------
# SpO2 ANALYSIS
# ---------------------------------------------------------

spo2 = df["SpO2(%)"]

valid_spo2 = spo2.dropna()

mean_spo2 = valid_spo2.mean()
min_spo2 = valid_spo2.min()
max_spo2 = valid_spo2.max()

missing_spo2 = spo2.isna().sum()
missing_spo2_percentage = (missing_spo2 / total_rows) * 100

print("\n" + "=" * 60)
print("SpO2 ANALYSIS")
print("=" * 60)

print(f"Mean SpO2       : {mean_spo2:.2f}%")
print(f"Minimum SpO2    : {min_spo2:.2f}%")
print(f"Maximum SpO2    : {max_spo2:.2f}%")
print(f"Missing samples : {missing_spo2}")
print(f"Missing (%)     : {missing_spo2_percentage:.2f}%")

# ---------------------------------------------------------
# PPG VALIDITY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PPG SIGNAL QUALITY")
print("=" * 60)

# Percentage of finite samples inside each 80-sample PPG segment
ppg_valid_fraction = df["signal_pleth"].apply(
    lambda x: np.isfinite(x).mean()
)

fully_valid_ppg = (ppg_valid_fraction == 1).sum()
partially_valid_ppg = (
    (ppg_valid_fraction > 0) &
    (ppg_valid_fraction < 1)
).sum()
invalid_ppg = (ppg_valid_fraction == 0).sum()

print(f"Fully valid PPG rows     : {fully_valid_ppg}")
print(f"Partially valid PPG rows : {partially_valid_ppg}")
print(f"Completely invalid PPG   : {invalid_ppg}")

print(
    f"Average PPG validity     : "
    f"{ppg_valid_fraction.mean() * 100:.2f}%"
)

# ---------------------------------------------------------
# EVENT COUNTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RESPIRATORY EVENT DISTRIBUTION")
print("=" * 60)

event_counts = df["event"].value_counts()

print(event_counts)

# ---------------------------------------------------------
# NORMAL VS ABNORMAL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NORMAL VS ABNORMAL")
print("=" * 60)

normal_count = (df["anomaly"] == False).sum()
abnormal_count = (df["anomaly"] == True).sum()

print(f"Normal records   : {normal_count}")
print(f"Abnormal records : {abnormal_count}")

print(
    f"Normal percentage   : "
    f"{normal_count / total_rows * 100:.2f}%"
)

print(
    f"Abnormal percentage : "
    f"{abnormal_count / total_rows * 100:.2f}%"
)

# ---------------------------------------------------------
# SpO2 COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SpO2: NORMAL VS ABNORMAL")
print("=" * 60)

normal_spo2 = df.loc[
    df["anomaly"] == False,
    "SpO2(%)"
].dropna()

abnormal_spo2 = df.loc[
    df["anomaly"] == True,
    "SpO2(%)"
].dropna()

print(
    f"Normal mean SpO2   : "
    f"{normal_spo2.mean():.2f}%"
)

print(
    f"Abnormal mean SpO2 : "
    f"{abnormal_spo2.mean():.2f}%"
)

print(
    f"Normal minimum     : "
    f"{normal_spo2.min():.2f}%"
)

print(
    f"Abnormal minimum   : "
    f"{abnormal_spo2.min():.2f}%"
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)