import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# FIND A USABLE EVENT
# ---------------------------------------------------------

print("\nSearching for an event with usable PPG and SpO2...")

# Look through patients separately so that windows
# never cross patient boundaries.
selected_patient = None
selected_event_index = None

for patient_id in df["patient"].unique():

    patient_df = df[df["patient"] == patient_id].reset_index(drop=True)

    event_indices = patient_df.index[
        patient_df["anomaly"] == True
    ]

    for event_index in event_indices:

        start = max(0, event_index - 30)
        end = min(len(patient_df), event_index + 30)

        segment = patient_df.iloc[start:end]

        # Check PPG validity
        ppg_valid_fraction = segment["signal_pleth"].apply(
            lambda x: np.isfinite(x).mean()
        ).mean()

        # Check SpO2 validity
        spo2_valid_fraction = segment["SpO2(%)"].notna().mean()

        # Select a reasonably good segment
        if (
            ppg_valid_fraction >= 0.80
            and spo2_valid_fraction >= 0.50
        ):

            selected_patient = patient_id
            selected_event_index = event_index

            print("Usable event found!")
            break

    if selected_patient is not None:
        break

# ---------------------------------------------------------
# CHECK RESULT
# ---------------------------------------------------------

if selected_patient is None:

    print("No suitable event was found.")

    raise RuntimeError(
        "Could not find an event with sufficient valid PPG and SpO2."
    )

# ---------------------------------------------------------
# SELECT PATIENT
# ---------------------------------------------------------

patient_df = df[
    df["patient"] == selected_patient
].copy()

patient_df = patient_df.reset_index(drop=True)

# ---------------------------------------------------------
# SELECT 60 SECOND WINDOW
# ---------------------------------------------------------

event_start = selected_event_index

start = max(0, event_start - 30)
end = min(len(patient_df), event_start + 30)

segment = patient_df.iloc[start:end].copy()

# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SELECTED SIGNAL SEGMENT")
print("=" * 60)

print("Patient:", selected_patient)

print(
    "Event:",
    patient_df.loc[event_start, "event"]
)

print(
    "Event timestamp:",
    patient_df.loc[
        event_start,
        "timestamp_datetime"
    ]
)

print(
    "Window start:",
    segment["timestamp_datetime"].iloc[0]
)

print(
    "Window end:",
    segment["timestamp_datetime"].iloc[-1]
)

# ---------------------------------------------------------
# VALIDITY INFORMATION
# ---------------------------------------------------------

ppg_valid_fraction = segment["signal_pleth"].apply(
    lambda x: np.isfinite(x).mean()
).mean()

spo2_valid_fraction = segment["SpO2(%)"].notna().mean()

print(
    f"PPG valid fraction: {ppg_valid_fraction * 100:.2f}%"
)

print(
    f"SpO2 valid fraction: {spo2_valid_fraction * 100:.2f}%"
)

# ---------------------------------------------------------
# PPG
# ---------------------------------------------------------

ppg = np.concatenate(
    segment["signal_pleth"].values
)

ppg_time = np.arange(len(ppg)) / 80

# ---------------------------------------------------------
# SpO2
# ---------------------------------------------------------

spo2 = segment["SpO2(%)"].values

time = np.arange(len(segment))

# ---------------------------------------------------------
# EVENT MASK
# ---------------------------------------------------------

event_mask = segment["anomaly"].fillna(False).values

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=False
)

# ---------------------------------------------------------
# PPG
# ---------------------------------------------------------

axes[0].plot(
    ppg_time,
    ppg,
    linewidth=0.8
)

axes[0].set_title(
    f"PPG Signal Around Respiratory Event - Patient {selected_patient}"
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("PPG Amplitude")

axes[0].grid(True)

# ---------------------------------------------------------
# SpO2
# ---------------------------------------------------------

axes[1].plot(
    time,
    spo2,
    marker=".",
    linewidth=1.2
)

axes[1].set_title("SpO₂ Signal")

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("SpO₂ (%)")

axes[1].grid(True)

# ---------------------------------------------------------
# EVENT REGION
# ---------------------------------------------------------

if np.any(event_mask):

    event_times = time[event_mask]

    event_start_time = event_times.min()
    event_end_time = event_times.max()

    for ax in axes:

        ax.axvspan(
            event_start_time,
            event_end_time + 1,
            alpha=0.25,
            label="Apnea/Hypopnea"
        )

        ax.legend()

# ---------------------------------------------------------
# FIGURE TITLE
# ---------------------------------------------------------

fig.suptitle(
    "OSASUD: PPG and SpO₂ Around an Annotated Respiratory Event",
    fontsize=14
)

plt.tight_layout()

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_dir = project_root / "results"

output_dir.mkdir(
    exist_ok=True
)

output_file = (
    output_dir /
    "ppg_spo2_event_visualization.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nFigure saved to:")
print(output_file)

plt.show()