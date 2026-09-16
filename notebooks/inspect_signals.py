import pandas as pd
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = project_root / "data" / "raw" / "dataset_OSAS.pickle"

print("=" * 60)
print("OSASUD SIGNAL STRUCTURE INSPECTION")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_pickle(dataset_path)

print("Dataset loaded successfully!")

# ---------------------------------------------------------
# Inspect one row
# ---------------------------------------------------------

row = df.iloc[0]

print("\n" + "=" * 60)
print("FIRST ROW INFORMATION")
print("=" * 60)

print("\nPatient:")
print(row["patient"])

print("\nTimestamp:")
print(row["timestamp_datetime"])

print("\nHR:")
print(row["HR(bpm)"])

print("\nSpO2:")
print(row["SpO2(%)"])

print("\nRespiratory Rate:")
print(row["RR(rpm)"])

print("\nEvent:")
print(row["event"])

print("\nAnomaly:")
print(row["anomaly"])

# ---------------------------------------------------------
# PPG
# ---------------------------------------------------------

ppg = row["signal_pleth"]

print("\n" + "=" * 60)
print("PPG INFORMATION")
print("=" * 60)

print("Python type:", type(ppg))
print("Number of samples:", len(ppg))

print("\nFirst 10 PPG samples:")
print(ppg[:10])

# ---------------------------------------------------------
# ECG
# ---------------------------------------------------------

ecg = row["signal_ecg_i"]

print("\n" + "=" * 60)
print("ECG INFORMATION")
print("=" * 60)

print("Python type:", type(ecg))
print("Number of samples:", len(ecg))

print("\nFirst 10 ECG samples:")
print(ecg[:10])

# ---------------------------------------------------------
# PSG signals
# ---------------------------------------------------------

psg_flow = row["PSG_Flow"]

print("\n" + "=" * 60)
print("PSG FLOW INFORMATION")
print("=" * 60)

print("Python type:", type(psg_flow))
print("Number of samples:", len(psg_flow))

print("\nFirst 10 PSG Flow samples:")
print(psg_flow[:10])

print("\n" + "=" * 60)
print("SIGNAL INSPECTION COMPLETE")
print("=" * 60)