import pandas as pd
from pathlib import Path

# Path to OSASUD dataset
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "raw" / "dataset_OSAS.pickle"

print("=" * 60)
print("OSASUD DATASET INSPECTION")
print("=" * 60)

# Check whether file exists
print("\nChecking dataset file...")

if dataset_path.exists():
    print("Dataset found!")
    print(f"File: {dataset_path}")
    print(f"Size: {dataset_path.stat().st_size / (1024**3):.2f} GB")
else:
    print("ERROR: Dataset file not found!")
    raise FileNotFoundError(dataset_path)

# Load dataset
print("\nLoading dataset...")
df = pd.read_pickle(dataset_path)

print("Dataset loaded successfully!")

# Basic information
print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# Column names
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# First few rows
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())

# Data types
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)