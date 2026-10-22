
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sleep Apnea Signal Explorer",
    page_icon="🫁",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "dashboard_records.parquet"
SIGNAL_PATH = ROOT / "data" / "processed" / "signal_windows.parquet"

st.title("Sleep Apnea Signal Explorer")
st.caption(
    "OSASUD dataset | PPG and SpO₂ | "
    "Research prototype, not a diagnostic tool"
)


@st.cache_data(show_spinner="Loading processed records...")
def load_data(path):
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading PPG sample...")
def load_signals(path):
    return pd.read_parquet(path)


if not DATA_PATH.exists():
    st.error("Dashboard data not found. Run notebooks/prepare_dashboard_data.py first.")
    st.stop()

df = load_data(str(DATA_PATH))

# Sidebar filters
st.sidebar.header("Dashboard filters")

patients = sorted(df["patient"].dropna().unique().tolist())
selected_patients = st.sidebar.multiselect(
    "Select patients",
    patients,
    default=patients,
)

events = sorted(df["event"].dropna().astype(str).unique().tolist())
selected_events = st.sidebar.multiselect(
    "Select event labels",
    events,
    default=events,
)

filtered = df[
    df["patient"].isin(selected_patients)
    & df["event"].astype(str).isin(selected_events)
]

# Dataset overview
st.subheader("Dataset overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Patients", f"{filtered['patient'].nunique():,}")
c3.metric(
    "Event-labelled records",
    f"{int(filtered['anomaly'].fillna(False).sum()):,}",
)
c4.metric(
    "Mean SpO₂",
    f"{filtered['SpO2(%)'].mean():.2f}%"
    if filtered["SpO2(%)"].notna().any()
    else "N/A",
)

st.caption(
    "Event-labelled records are annotated records, not a count of distinct "
    "clinical apnea episodes."
)

# Event distribution
st.subheader("Respiratory-event label distribution")
st.bar_chart(filtered["event"].astype(str).value_counts())

# SpO2 analysis
st.subheader("SpO₂ analysis")
valid_spo2 = filtered["SpO2(%)"].dropna()

if not valid_spo2.empty:
    a, b, c = st.columns(3)
    a.metric("Mean SpO₂", f"{valid_spo2.mean():.2f}%")
    b.metric("Minimum SpO₂", f"{valid_spo2.min():.2f}%")
    c.metric("Missing SpO₂", f"{filtered['SpO2(%)'].isna().sum():,}")

    st.write("Average SpO₂ by event label")
    st.bar_chart(
        filtered.groupby(filtered["event"].astype(str))["SpO2(%)"].mean()
    )
else:
    st.warning("No valid SpO₂ measurements in the selected records.")

# PPG signal viewer
st.divider()
st.header("PPG signal viewer")
st.write(
    "Explore an individual one-second PPG recording and its associated "
    "SpO₂ measurement and respiratory-event annotation."
)

if not SIGNAL_PATH.exists():
    st.warning(
        "PPG sample not found. Run notebooks/prepare_signal_windows.py first."
    )
else:
    signals = load_signals(str(SIGNAL_PATH))

    signal_patients = sorted(signals["patient"].dropna().unique().tolist())
    chosen_patient = st.selectbox(
        "Choose patient",
        signal_patients,
        key="signal_patient",
    )

    patient_records = signals[signals["patient"] == chosen_patient].copy()
    signal_events = sorted(patient_records["event"].dropna().astype(str).unique())

    chosen_event = st.selectbox(
        "Choose respiratory-event label",
        signal_events,
        key="signal_event",
    )

    candidates = patient_records[
        patient_records["event"].astype(str) == chosen_event
    ].reset_index(drop=True)

    if not candidates.empty:
        record_number = st.number_input(
            "Record number",
            min_value=1,
            max_value=len(candidates),
            value=1,
            step=1,
        )

        record = candidates.iloc[int(record_number) - 1]
        ppg = np.asarray(record["ppg"], dtype=float)
        ppg = ppg[np.isfinite(ppg)]

        m1, m2, m3 = st.columns(3)
        m1.metric("Patient", str(record["patient"]))
        m2.metric(
            "SpO₂",
            f"{record['SpO2(%)']:.2f}%"
            if pd.notna(record["SpO2(%)"])
            else "Unavailable",
        )
        m3.metric("PPG valid fraction", f"{record['ppg_valid_fraction']:.1%}")

        st.write(f"**Event label:** {record['event']}")
        st.write(f"**Timestamp:** {record['timestamp_datetime']}")

        if len(ppg) > 1:
            # The stored PPG arrays contain 80 samples per second.
            time_seconds = np.arange(len(ppg)) / 80.0

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time_seconds, ppg, linewidth=1.2)
            ax.set_title("Photoplethysmography (PPG) waveform")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Signal amplitude (recorded units)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "This plot shows one stored PPG record, approximately one "
                "second long at 80 samples per second. It is not a continuous "
                "multi-second apnea episode."
            )
        else:
            st.warning("This record does not contain enough valid PPG samples.")
    else:
        st.info("No records match the selected patient and event label.")

# Record table
st.divider()
st.subheader("Record explorer")
st.dataframe(filtered.head(500), width="stretch")

st.caption(
    "This version explores dataset measurements, annotations, and individual "
    "PPG records. It does not yet generate ML predictions or SHAP explanations."
)


# ============================================================
# STEP 12: MODEL PREDICTION AND SHAP EXPLANATION
# ============================================================

import joblib
import shap
import pandas as pd
import streamlit as st

st.divider()
st.header("Experimental XGBoost Prediction & Explainability")

st.warning(
    "Research prototype only. Predictions are experimental estimates, "
    "not medical diagnoses. The current model was trained on records "
    "from one patient and has not been clinically validated."
)

ROOT_DIR = Path(__file__).resolve().parent
ML_DATA_PATH = ROOT_DIR / "data" / "processed" / "ml_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_weighted_demo.joblib"

if not ML_DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Required model or ML dataset is missing. "
        "Run the model preparation and weighted training scripts first."
    )
else:
    ml_df = pd.read_csv(ML_DATA_PATH)
    model_bundle = joblib.load(MODEL_PATH)

    prediction_model = model_bundle["model"]
    model_features = model_bundle["feature_columns"]
    prediction_threshold = float(model_bundle["threshold"])

    # Keep only rows with all required input features
    ml_df = ml_df.dropna(subset=model_features).reset_index(drop=True)

    st.caption(
        f"Available records: {len(ml_df):,} | "
        f"Decision threshold: {prediction_threshold:.2f}"
    )

    if ml_df.empty:
        st.error("No records have complete values for the model features.")
    else:
        selected_row_number = st.number_input(
            "Select dataset record number",
            min_value=1,
            max_value=len(ml_df),
            value=1,
            step=1,
            key="xgb_record_number",
        )

        selected_row = ml_df.iloc[[int(selected_row_number) - 1]]
        input_features = selected_row[model_features]

        # Model prediction
        event_probability = float(
            prediction_model.predict_proba(input_features)[0, 1]
        )
        predicted_class = int(
            event_probability >= prediction_threshold
        )

        st.subheader("Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Event probability",
                f"{event_probability:.1%}",
            )

        with col2:
            st.metric(
                "Decision threshold",
                f"{prediction_threshold:.2f}",
            )

        with col3:
            st.metric(
                "Predicted class",
                "EVENT-LABELLED" if predicted_class == 1 else "NONE",
            )

        # Display available record context
        context_columns = [
            col for col in [
                "patient", "timestamp_datetime", "event",
                "spo2", "target"
            ]
            if col in selected_row.columns
        ]

        st.subheader("Selected Record")
        st.dataframe(
            selected_row[context_columns].reset_index(drop=True),
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "EVENT-LABELLED means the model estimates that this record "
            "belongs to the event-labelled class. It does not identify "
            "a confirmed clinical event."
        )

        # Explain this particular prediction with SHAP
        st.subheader("Why did the model make this prediction?")

        try:
            explainer = shap.TreeExplainer(prediction_model)
            shap_explanation = explainer(input_features)
            shap_values = shap_explanation.values

            # Binary XGBoost models normally return one SHAP value per feature.
            if shap_values.ndim == 3:
                shap_values = shap_values[0, :, -1]
            else:
                shap_values = shap_values[0]

            explanation_df = pd.DataFrame({
                "Feature": model_features,
                "Feature value": input_features.iloc[0].to_numpy(),
                "SHAP contribution": shap_values,
            })

            explanation_df["Absolute contribution"] = (
                explanation_df["SHAP contribution"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute contribution",
                ascending=False,
            )

            st.caption(
                "Positive SHAP contributions push the model output "
                "toward the event-labelled class; negative contributions "
                "push it away. These values explain the model, not "
                "clinical causation."
            )

            top_features = explanation_df.head(8).sort_values(
                "SHAP contribution"
            )

            st.bar_chart(
                top_features.set_index("Feature")[["SHAP contribution"]]
            )

            st.dataframe(
                explanation_df.drop(
                    columns=["Absolute contribution"]
                ).reset_index(drop=True),
                hide_index=True,
                width="stretch",
            )

        except Exception as exc:
            st.error(f"Could not calculate this record's SHAP explanation: {exc}")
