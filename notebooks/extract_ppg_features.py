
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "signal_windows.parquet"
OUTPUT = ROOT / "data" / "processed" / "ppg_features.csv"
FIGURE = ROOT / "results" / "ppg_feature_distributions.png"

print("PPG FEATURE EXTRACTION")
print("-" * 40)

if not INPUT.exists():
    raise FileNotFoundError(f"Input sample not found: {INPUT}")

df = pd.read_parquet(INPUT)
print(f"Loaded {len(df):,} records")

features = []

for _, row in df.iterrows():
    try:
        signal = np.asarray(row["ppg"], dtype=float).ravel()
    except (TypeError, ValueError):
        signal = np.array([], dtype=float)

    signal = signal[np.isfinite(signal)]

    record = {
        "source_index": row["source_index"],
        "patient": row["patient"],
        "timestamp_datetime": row["timestamp_datetime"],
        "event": str(row["event"]),
        # Target for a future binary baseline: NONE vs annotated event.
        "target_event": int(str(row["event"]) != "NONE"),
        "spo2": pd.to_numeric(row["SpO2(%)"], errors="coerce"),
    }

    # Do not calculate waveform features from too few samples.
    if len(signal) < 64:
        for name in [
            "ppg_mean", "ppg_std", "ppg_min", "ppg_max",
            "ppg_range", "ppg_median", "ppg_iqr",
            "ppg_rms", "ppg_mean_abs_diff", "ppg_slope",
        ]:
            record[name] = np.nan
    else:
        record.update({
            "ppg_mean": float(np.mean(signal)),
            "ppg_std": float(np.std(signal)),
            "ppg_min": float(np.min(signal)),
            "ppg_max": float(np.max(signal)),
            "ppg_range": float(np.ptp(signal)),
            "ppg_median": float(np.median(signal)),
            "ppg_iqr": float(
                np.percentile(signal, 75) - np.percentile(signal, 25)
            ),
            "ppg_rms": float(np.sqrt(np.mean(signal ** 2))),
            "ppg_mean_abs_diff": float(np.mean(np.abs(np.diff(signal)))),
            "ppg_slope": float(np.polyfit(
                np.arange(len(signal)), signal, 1
            )[0]),
        })

    features.append(record)

result = pd.DataFrame(features)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
FIGURE.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(OUTPUT, index=False)

print("\nFEATURE EXTRACTION SUMMARY")
print("Records:", len(result))
print("Patients:", result["patient"].nunique())
print("Features created:", 10)
print("\nEvent-label counts:")
print(result["event"].value_counts().to_string())
print("\nMissing SpO2 values:", result["spo2"].isna().sum())
print("Saved feature table:", OUTPUT)

# Plot distributions of selected features.
plot_columns = [
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_mean_abs_diff",
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

for ax, column in zip(axes.flat, plot_columns):
    values = result[column].dropna()
    ax.hist(values, bins=30)
    ax.set_title(column.replace("_", " ").title())
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Record count")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Distributions of Extracted PPG Features")
fig.tight_layout()
fig.savefig(FIGURE, dpi=200)
plt.close(fig)

print("Feature distribution figure:", FIGURE)
print("\nNOTE: These features are exploratory, not clinically validated.")
print("The current sample is not suitable for patient-independent evaluation.")
