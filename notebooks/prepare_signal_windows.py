
from pathlib import Path
import gc

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data" / "raw" / "dataset_OSAS.pickle"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT = OUTPUT_DIR / "signal_windows.parquet"

# Collect at most 3,000 usable one-second records.
# These are individual records, not necessarily consecutive windows.
MAX_RECORDS = 3000
MIN_PPG_VALID_FRACTION = 0.80

REQUIRED_COLUMNS = [
    "patient",
    "timestamp_datetime",
    "SpO2(%)",
    "event",
    "anomaly",
    "signal_pleth",
]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("OSASUD SIGNAL SAMPLE PREPARATION")
    print("=" * 60)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    if OUTPUT.exists():
        print(f"\nOutput already exists:\n{OUTPUT}")
        print("No changes made. Rename or remove the output file")
        print("manually if you want to rebuild the sample.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource size: {SOURCE.stat().st_size / (1024**3):.2f} GB")
    print("Loading original dataset. Please wait...")

    df = pd.read_pickle(SOURCE)

    print(f"Dataset loaded: {len(df):,} records")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Inspect records in dataset order. Stop when enough usable records
    # have been collected instead of processing all remaining rows.
    print("\nSearching for usable PPG and SpO2 records...")

    records = []

    for position, (idx, row) in enumerate(df.iterrows()):
        ppg = row["signal_pleth"]
        spo2 = row["SpO2(%)"]

        if not isinstance(ppg, (np.ndarray, list, tuple)):
            continue

        ppg = np.asarray(ppg, dtype=np.float32)

        if ppg.size == 0 or pd.isna(spo2):
            continue

        valid_mask = np.isfinite(ppg)
        valid_fraction = float(valid_mask.mean())

        if valid_fraction < MIN_PPG_VALID_FRACTION:
            continue

        records.append({
            "source_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "patient": str(row["patient"]),
            "timestamp_datetime": row["timestamp_datetime"],
            "SpO2(%)": float(spo2),
            "event": str(row["event"]),
            "anomaly": bool(row["anomaly"]),
            "ppg": ppg.tolist(),
            "ppg_valid_fraction": valid_fraction,
        })

        if len(records) % 250 == 0:
            print(f"Collected {len(records):,} usable records...")

        if len(records) >= MAX_RECORDS:
            break

    print(f"\nUsable records collected: {len(records):,}")

    # Free the large source DataFrame before saving the smaller sample.
    del df
    gc.collect()

    if not records:
        raise RuntimeError(
            "No usable records found. Check PPG and SpO2 values."
        )

    result = pd.DataFrame(records)

    print("Saving sample to Parquet...")
    result.to_parquet(OUTPUT, index=False)

    print("Verifying saved sample...")
    check = pd.read_parquet(OUTPUT)

    if len(check) != len(result):
        raise RuntimeError("Verification failed: row count changed.")

    if check["ppg"].isna().any():
        raise RuntimeError("Verification failed: missing PPG arrays found.")

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Records saved     : {len(check):,}")
    print(f"Patients          : {check['patient'].nunique()}")
    print(f"Event labels      : {sorted(check['event'].unique().tolist())}")
    print(f"Output size       : {OUTPUT.stat().st_size / (1024**2):.2f} MB")
    print(f"Output file       : {OUTPUT}")


if __name__ == "__main__":
    main()
