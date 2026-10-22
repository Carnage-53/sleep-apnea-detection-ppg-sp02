
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "signal_windows.parquet"
OUTPUT = ROOT / "data" / "processed" / "signal_quality_report.csv"
FIGURE = ROOT / "results" / "ppg_signal_quality_assessment.png"

print("PPG SIGNAL QUALITY ASSESSMENT")
print("-" * 40)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Signal sample not found: {INPUT}\n"
        "Run notebooks/prepare_signal_windows.py first."
    )

df = pd.read_parquet(INPUT)
print(f"Loaded {len(df):,} PPG records")

results = []

for _, row in df.iterrows():
    try:
        signal = np.asarray(row["ppg"], dtype=float).ravel()
    except (TypeError, ValueError):
        signal = np.array([], dtype=float)

    finite = np.isfinite(signal)
    valid_fraction = float(finite.mean()) if signal.size else 0.0
    valid_signal = signal[finite]

    if valid_signal.size:
        signal_std = float(np.std(valid_signal))
        signal_min = float(np.min(valid_signal))
        signal_max = float(np.max(valid_signal))
        zero_fraction = float(np.mean(valid_signal == 0))
    else:
        signal_std = np.nan
        signal_min = np.nan
        signal_max = np.nan
        zero_fraction = np.nan

    # Screening rules for this exploratory project, not clinical criteria.
    flags = []

    if signal.size != 80:
        flags.append("unexpected_length")

    if valid_fraction < 0.95:
        flags.append("missing_or_invalid_samples")

    if valid_signal.size and signal_std < 2:
        flags.append("near_constant_signal")

    if valid_signal.size and zero_fraction > 0.10:
        flags.append("many_zero_samples")

    results.append({
        "source_index": row["source_index"],
        "patient": row["patient"],
        "timestamp_datetime": row["timestamp_datetime"],
        "event": row["event"],
        "spo2": row["SpO2(%)"],
        "valid_fraction": valid_fraction,
        "signal_std": signal_std,
        "signal_min": signal_min,
        "signal_max": signal_max,
        "zero_fraction": zero_fraction,
        "quality_flags": ";".join(flags) if flags else "no_basic_flags",
    })

report = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
FIGURE.parent.mkdir(parents=True, exist_ok=True)

report.to_csv(OUTPUT, index=False)

print("\nQUALITY SUMMARY")
print("Records:", len(report))
print("Records with no basic flags:",
      (report["quality_flags"] == "no_basic_flags").sum())
print("Records with quality flags:",
      (report["quality_flags"] != "no_basic_flags").sum())
print("Mean valid fraction:",
      f"{report['valid_fraction'].mean():.2%}")
print("Report saved:", OUTPUT)

# Create a summary figure for the report/PPT.
flag_counts = (
    report["quality_flags"]
    .eq("no_basic_flags")
    .map({True: "No basic flags", False: "Flagged"})
    .value_counts()
    .reindex(["No basic flags", "Flagged"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(8, 5))
flag_counts.plot(kind="bar", ax=ax)
ax.set_title("Basic PPG Signal Quality Screening")
ax.set_xlabel("Screening category")
ax.set_ylabel("Number of records")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE, dpi=200)
plt.close(fig)

print("Figure saved:", FIGURE)
print("\nNOTE: These are exploratory screening rules,")
print("not clinically validated signal-quality criteria.")
