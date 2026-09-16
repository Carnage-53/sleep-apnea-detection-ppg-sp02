import pandas as pd
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
# CREATE RESULTS DIRECTORY
# ---------------------------------------------------------

results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# =========================================================
# FIGURE 1 — RESPIRATORY EVENT DISTRIBUTION
# =========================================================

event_counts = df["event"].value_counts()

# Put normal event first
event_order = [
    "NONE",
    "HYPOPNEA",
    "APNEA-OBSTRUCTIVE",
    "APNEA-CENTRAL",
    "APNEA-MIXED"
]

event_counts = event_counts.reindex(event_order)

plt.figure(figsize=(10, 6))

plt.bar(
    event_counts.index,
    event_counts.values
)

plt.title("OSASUD Respiratory Event Distribution")
plt.xlabel("Event Type")
plt.ylabel("Number of Records")

plt.xticks(rotation=20)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

event_plot = results_dir / "respiratory_event_distribution.png"

plt.savefig(
    event_plot,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(event_plot)

plt.show()

# =========================================================
# FIGURE 2 — NORMAL VS ABNORMAL SpO2
# =========================================================

normal_spo2 = df.loc[
    df["anomaly"] == False,
    "SpO2(%)"
].dropna()

abnormal_spo2 = df.loc[
    df["anomaly"] == True,
    "SpO2(%)"
].dropna()

plt.figure(figsize=(8, 6))

plt.boxplot(
    [normal_spo2, abnormal_spo2],
    tick_labels=["Normal", "Apnea/Hypopnea"]
)

plt.title("SpO₂ Distribution: Normal vs Apnea/Hypopnea")
plt.ylabel("SpO₂ (%)")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

spo2_plot = results_dir / "spo2_normal_vs_abnormal.png"

plt.savefig(
    spo2_plot,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(spo2_plot)

plt.show()

print("\n" + "=" * 60)
print("RESULT FIGURES COMPLETE")
print("=" * 60)