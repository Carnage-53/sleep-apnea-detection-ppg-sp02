
from pathlib import Path
import gc
import sys

import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_OSAS.pickle"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "dashboard_records.parquet"

# Only keep columns needed for the first dashboard and ML baseline.
KEEP_COLUMNS = [
    "patient",
    "timestamp_datetime",
    "HR(bpm)",
    "SpO2(%)",
    "RR(rpm)",
    "event",
    "anomaly",
]

print("=" * 60)
print("OSASUD DASHBOARD DATA PREPARATION")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print(f"Source file: {DATASET_PATH}")
print(f"Source size: {DATASET_PATH.stat().st_size / (1024**3):.2f} GB")
print(f"Output file: {OUTPUT_PATH}")

if OUTPUT_PATH.exists():
    print("\nA processed file already exists.")
    print("Delete or rename it manually only if you want to rebuild it.")
    sys.exit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading source dataset. This may take several minutes...")
df = pd.read_pickle(DATASET_PATH)

print("Dataset loaded.")
print(f"Original shape: {df.shape}")

missing_columns = [col for col in KEEP_COLUMNS if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("\nSelecting dashboard columns...")
dashboard_df = df[KEEP_COLUMNS].copy()

print("Converting event labels to compact strings...")
dashboard_df["event"] = dashboard_df["event"].astype("category")

print("Saving processed data as Parquet...")
try:
    dashboard_df.to_parquet(OUTPUT_PATH, index=False)
except ImportError as exc:
    raise RuntimeError(
        "Parquet support is missing. Install it with: "
        "python -m pip install pyarrow"
    ) from exc

print("\nVerifying saved file...")
check_df = pd.read_parquet(OUTPUT_PATH)

if len(check_df) != len(df):
    raise RuntimeError("Verification failed: row count changed.")

print(f"Verified rows: {len(check_df):,}")
print(f"Verified columns: {len(check_df.columns)}")
print(f"Processed size: {OUTPUT_PATH.stat().st_size / (1024**2):.2f} MB")

del check_df, dashboard_df, df
gc.collect()

print("\nPreparation complete.")
print(f"Saved to: {OUTPUT_PATH}")
