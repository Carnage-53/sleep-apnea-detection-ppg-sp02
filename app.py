
from pathlib import Path
import ast
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SleepSense AI | Research Platform",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "xgboost_weighted_demo.joblib"

DASHBOARD_PATH = PROCESSED / "dashboard_records.parquet"
SIGNALS_PATH = PROCESSED / "signal_windows.parquet"
ML_PATH = PROCESSED / "ml_dataset.csv"
PREDICTIONS_PATH = RESULTS / "xgboost_weighted_test_predictions.csv"
IMPORTANCE_PATH = RESULTS / "shap_feature_importance.csv"
THRESHOLD_PATH = RESULTS / "xgboost_threshold_comparison.csv"
CONFUSION_PATH = RESULTS / "xgboost_weighted_confusion_matrix.png"

EVENT_ORDER = [
    "NONE",
    "HYPOPNEA",
    "APNEA-CENTRAL",
    "APNEA-OBSTRUCTIVE",
    "APNEA-MIXED",
]


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1220;
        color: #E5EDF7;
    }
    [data-testid="stSidebar"] {
        background-color: #101B2B;
        border-right: 1px solid #26364C;
    }
    [data-testid="stMetric"] {
        background: #121D2E;
        border: 1px solid #26364C;
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8;
    }
    .hero {
        padding: 25px 28px;
        border: 1px solid #26364C;
        border-radius: 18px;
        background: linear-gradient(120deg, #13263A, #102B32);
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .hero p {
        color: #B8C7D9;
        margin-bottom: 0;
    }
    .tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        background: #173D42;
        color: #7DE8D7;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }
    .notice {
        padding: 13px 16px;
        background: #352B1B;
        border-left: 4px solid #FBBF24;
        border-radius: 7px;
        color: #FDE7B0;
        margin: 12px 0;
    }
    .small-muted {
        color: #94A3B8;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def file_exists(path):
    return path.exists() and path.is_file()


def modified_time(path):
    return path.stat().st_mtime if path.exists() else 0


def format_number(value, digits=2, suffix=""):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if not np.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def clean_event_column(series):
    return series.fillna("UNKNOWN").astype(str).str.strip().str.upper()


def get_signal_array(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        array = np.asarray(value, dtype=float).reshape(-1)
        return array[np.isfinite(array)]
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return np.array([], dtype=float)


def show_missing_files(paths):
    missing = [
        str(path.relative_to(ROOT))
        for path in paths
        if not file_exists(path)
    ]
    if missing:
        st.error("Required project files are missing:")
        for item in missing:
            st.code(item)
        st.info(
            "Check the filenames and folders. For deployment, ensure required "
            "files are committed to GitHub."
        )
        st.stop()


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner="Loading research dataset...")
def load_dashboard(path, mtime):
    df = pd.read_parquet(path)

    if "timestamp_datetime" in df.columns:
        df["timestamp_datetime"] = pd.to_datetime(
            df["timestamp_datetime"], errors="coerce"
        )
    if "patient" in df.columns:
        df["patient"] = df["patient"].astype(str)
    if "event" in df.columns:
        df["event"] = clean_event_column(df["event"])

    for column in ["HR(bpm)", "SpO2(%)", "RR(rpm)"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(show_spinner="Loading PPG windows...")
def load_signals(path, mtime):
    df = pd.read_parquet(path)

    if "timestamp_datetime" in df.columns:
        df["timestamp_datetime"] = pd.to_datetime(
            df["timestamp_datetime"], errors="coerce"
        )
    if "patient" in df.columns:
        df["patient"] = df["patient"].astype(str)
    if "event" in df.columns:
        df["event"] = clean_event_column(df["event"])

    return df


@st.cache_data(show_spinner="Loading model feature data...")
def load_ml_data(path, mtime):
    df = pd.read_csv(path)

    if "patient" in df.columns:
        df["patient"] = df["patient"].astype(str)
    if "event" in df.columns:
        df["event"] = clean_event_column(df["event"])

    return df


@st.cache_resource(show_spinner="Loading saved model...")
def load_model(path, mtime):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_csv(path, mtime):
    return pd.read_csv(path)


# ============================================================
# CHECK AND LOAD FILES
# ============================================================
show_missing_files([
    DASHBOARD_PATH,
    SIGNALS_PATH,
    ML_PATH,
])

try:
    records = load_dashboard(
        str(DASHBOARD_PATH), modified_time(DASHBOARD_PATH)
    )
    signals = load_signals(
        str(SIGNALS_PATH), modified_time(SIGNALS_PATH)
    )
    ml_data = load_ml_data(
        str(ML_PATH), modified_time(ML_PATH)
    )
except Exception as exc:
    st.error("Could not load the project datasets.")
    st.exception(exc)
    st.stop()

if records.empty or "event" not in records.columns:
    st.error("The dashboard dataset is empty or missing the event column.")
    st.stop()

model_bundle = None
model_error = None

if file_exists(MODEL_PATH):
    try:
        model_bundle = load_model(
            str(MODEL_PATH), modified_time(MODEL_PATH)
        )
    except Exception as exc:
        model_error = str(exc)

predictions = None
importance = None
threshold_table = None

for path, variable in [
    (PREDICTIONS_PATH, "predictions"),
    (IMPORTANCE_PATH, "importance"),
    (THRESHOLD_PATH, "threshold_table"),
]:
    if file_exists(path):
        try:
            value = load_csv(str(path), modified_time(path))
            if variable == "predictions":
                predictions = value
            elif variable == "importance":
                importance = value
            else:
                threshold_table = value
        except Exception:
            pass


# ============================================================
# SUMMARY STATISTICS
# ============================================================
total_records = len(records)
patient_count = (
    records["patient"].nunique()
    if "patient" in records.columns else 0
)
event_counts = records["event"].value_counts()

none_count = int((records["event"] == "NONE").sum())
event_count = int(
    (
        (records["event"] != "NONE")
        & (records["event"] != "UNKNOWN")
    ).sum()
)

spo2_mean = (
    safe_mean(records["SpO2(%)"])
    if "SpO2(%)" in records.columns else None
)
hr_mean = (
    safe_mean(records["HR(bpm)"])
    if "HR(bpm)" in records.columns else None
)
rr_mean = (
    safe_mean(records["RR(rpm)"])
    if "RR(rpm)" in records.columns else None
)

model_threshold = 0.5
if isinstance(model_bundle, dict):
    try:
        model_threshold = float(model_bundle.get("threshold", 0.5))
    except (ValueError, TypeError):
        model_threshold = 0.5


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🌙 SleepSense AI")
    st.caption("Sleep Apnea Research Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Patient Explorer",
            "Signal Lab",
            "AI & Explainability",
            "Evaluation & Reports",
        ],
    )

    st.divider()
    st.markdown("**Dataset summary**")
    st.write(f"Records: {total_records:,}")
    st.write(f"Patients: {patient_count:,}")
    st.write(f"PPG windows: {len(signals):,}")
    st.divider()
    st.caption("Research prototype • Not for clinical diagnosis")


# ============================================================
# APP HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="tag">RESEARCH • SIGNAL PROCESSING • EXPLAINABLE AI</span>
        <h1>🌙 SleepSense AI</h1>
        <p>
        A multi-modal research platform for exploring PPG, SpO₂ and
        respiratory-event annotations, with experimental machine learning
        and explainability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
    <b>Important research limitation:</b> The current demonstration model
    was developed using records from one patient and a random record-level
    split. It has not been clinically validated and must not be used to
    diagnose, rule out or treat sleep apnea. Model probabilities are
    experimental outputs, not clinical risk estimates.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. OVERVIEW
# ============================================================
if page == "Overview":
    st.subheader("Research Overview")
    st.write(
        "Explore the dataset, annotated respiratory-event distribution, "
        "and available physiological measurements."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{total_records:,}")
    c2.metric("Unique patients", f"{patient_count:,}")
    c3.metric("Event-labelled records", f"{event_count:,}")
    c4.metric("PPG windows", f"{len(signals):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Mean SpO₂", format_number(spo2_mean, suffix=" %"))
    c6.metric("Mean heart rate", format_number(hr_mean, suffix=" bpm"))
    c7.metric("Mean respiratory rate", format_number(rr_mean, suffix=" rpm"))

    st.divider()
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("### Respiratory event annotations")
        counts = event_counts.sort_values()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        counts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of records")
        ax.set_ylabel("Event label")
        ax.set_title("Record counts by event label")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Counts represent annotated records, not unique clinical events."
        )

    with right:
        st.markdown("### Annotation summary")
        summary = (
            records["event"]
            .value_counts()
            .rename_axis("Event label")
            .reset_index(name="Record count")
        )
        summary["Percentage"] = (
            summary["Record count"] / max(total_records, 1) * 100
        ).round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("### Signal-window coverage")
        st.write(f"Stored PPG windows: **{len(signals):,}**")
        if "patient" in signals.columns:
            st.write(
                f"Patients represented: "
                f"**{signals['patient'].nunique():,}**"
            )
        st.caption(
            "PPG windows are a subset of the dataset and should not be "
            "interpreted as a continuous recording."
        )

    st.divider()
    st.markdown("### Research workflow")
    cols = st.columns(5)
    stages = [
        ("01", "Dataset", "Inspect records"),
        ("02", "Signals", "Explore PPG and SpO₂"),
        ("03", "Features", "Numerical feature extraction"),
        ("04", "Model", "XGBoost classification"),
        ("05", "Explain", "SHAP interpretation"),
    ]
    for col, (number, title, description) in zip(cols, stages):
        with col:
            st.markdown(f"**{number} · {title}**")
            st.caption(description)


# ============================================================
# 2. PATIENT EXPLORER
# ============================================================
elif page == "Patient Explorer":
    st.subheader("Patient Explorer")
    st.write(
        "Select a patient, filter annotated records, and explore physiological "
        "trends. These visualizations are descriptive, not diagnostic."
    )

    if "patient" not in records.columns:
        st.error("The dataset has no patient identifier.")
        st.stop()

    patients = sorted(records["patient"].dropna().unique().tolist())
    if not patients:
        st.warning("No patient identifiers were found.")
        st.stop()

    selected_patient = st.selectbox("Select patient", patients)
    patient_df = records[records["patient"] == selected_patient].copy()

    events = sorted(patient_df["event"].dropna().unique().tolist())
    selected_events = st.multiselect(
        "Filter event annotations",
        options=events,
        default=events,
    )
    filtered = patient_df[patient_df["event"].isin(selected_events)].copy()

    if "timestamp_datetime" in filtered.columns:
        valid_times = filtered["timestamp_datetime"].dropna()
        if not valid_times.empty:
            st.caption(
                f"Available time range: {valid_times.min()} to "
                f"{valid_times.max()}"
            )

    a, b, c = st.columns(3)
    a.metric("Patient records", f"{len(patient_df):,}")
    b.metric("Filtered records", f"{len(filtered):,}")
    c.metric(
        "Event-labelled records",
        f"{int((filtered['event'] != 'NONE').sum()):,}",
    )

    if filtered.empty:
        st.info("No records match the selected filters.")
    else:
        if "timestamp_datetime" in filtered.columns:
            plot_df = filtered.sort_values("timestamp_datetime").dropna(
                subset=["timestamp_datetime"]
            )

            if len(plot_df) > 15000:
                indices = np.linspace(
                    0, len(plot_df) - 1, 15000
                ).astype(int)
                plot_df = plot_df.iloc[indices]

            for column, label in [
                ("SpO2(%)", "SpO₂ (%)"),
                ("HR(bpm)", "Heart rate (bpm)"),
                ("RR(rpm)", "Respiratory rate (rpm)"),
            ]:
                if column in plot_df.columns:
                    st.markdown(f"### {label}")
                    chart = plot_df[["timestamp_datetime", column]].dropna()
                    if not chart.empty:
                        st.line_chart(
                            chart.set_index("timestamp_datetime"),
                            height=230,
                        )
                    else:
                        st.info(f"No valid {label} values are available.")
        else:
            st.info("Timestamps are unavailable for time-series plots.")

        st.markdown("### Event distribution")
        st.bar_chart(filtered["event"].value_counts())

        st.markdown("### Record explorer")
        st.caption("Showing up to 500 rows.")
        st.dataframe(
            filtered.head(500),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download filtered patient records (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"patient_{selected_patient}_records.csv",
            mime="text/csv",
        )


# ============================================================
# 3. SIGNAL LAB
# ============================================================
elif page == "Signal Lab":
    st.subheader("Signal Lab")
    st.write(
        "Inspect stored PPG waveform samples and their associated annotations."
    )

    if signals.empty:
        st.warning("No PPG windows are available.")
        st.stop()

    signal_df = signals.copy()

    if "patient" in signal_df.columns:
        patients = sorted(signal_df["patient"].dropna().unique().tolist())
        if patients:
            chosen_patient = st.selectbox(
                "Patient in PPG sample", patients
            )
            signal_df = signal_df[
                signal_df["patient"] == chosen_patient
            ]

    if "event" in signal_df.columns:
        events = sorted(signal_df["event"].dropna().unique().tolist())
        chosen_event = st.selectbox(
            "Filter PPG windows",
            ["All events"] + events,
        )
        if chosen_event != "All events":
            signal_df = signal_df[signal_df["event"] == chosen_event]

    if signal_df.empty:
        st.info("No windows match the selected filters.")
        st.stop()

    indices = signal_df.index.tolist()
    selected_index = st.selectbox(
        "Select stored signal window",
        indices,
        format_func=lambda idx: (
            f"Row {idx} | "
            f"{signal_df.loc[idx, 'event'] if 'event' in signal_df.columns else 'Unknown'}"
            f" | SpO₂: "
            f"{signal_df.loc[idx, 'SpO2(%)'] if 'SpO2(%)' in signal_df.columns else 'N/A'}"
        ),
    )

    selected_row = signal_df.loc[selected_index]

    if "ppg" not in signal_df.columns:
        st.error("The PPG window file has no 'ppg' column.")
        st.stop()

    ppg = get_signal_array(selected_row["ppg"])

    if ppg.size == 0:
        st.warning("The selected row does not contain a readable PPG waveform.")
    else:
        st.markdown("### PPG waveform")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(len(ppg)), ppg, linewidth=1.3)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Stored amplitude")
        ax.set_title("Photoplethysmography sample window")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The horizontal axis is sample index, not seconds, unless the "
            "sampling frequency is known."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Stored samples", f"{len(ppg):,}")
        m2.metric("Amplitude range", format_number(np.ptp(ppg)))
        m3.metric("Amplitude standard deviation", format_number(np.std(ppg)))

        metadata = {}
        for column in [
            "patient", "timestamp_datetime", "SpO2(%)",
            "event", "anomaly", "ppg_valid_fraction", "source_index",
        ]:
            if column in selected_row.index:
                value = selected_row[column]
                if isinstance(value, np.generic):
                    value = value.item()
                metadata[column] = str(value) if pd.isna(value) else value

        st.markdown("### Associated record metadata")
        st.json(metadata)

        st.markdown("### Basic signal-quality checks")
        checks = {
            "Contains numeric samples": ppg.size > 0,
            "All retained samples are finite": bool(np.isfinite(ppg).all()),
            "Signal is not constant": bool(np.ptp(ppg) > 0),
            "At least 20 samples available": ppg.size >= 20,
        }
        for label, passed in checks.items():
            st.write(f"{'✅' if passed else '⚠️'} {label}")

        st.caption(
            "These basic checks are not a validated signal-quality index."
        )

    st.info(
        "The stored PPG windows are a sampled subset from one patient and are "
        "not guaranteed to form a continuous recording. Avoid population-level "
        "conclusions from these windows."
    )


# ============================================================
# 4. AI & EXPLAINABILITY
# ============================================================
elif page == "AI & Explainability":
    st.subheader("AI & Explainability")
    st.write(
        "Generate a prediction from an existing feature row and inspect "
        "global and record-level explanations."
    )

    st.warning(
        "The model distinguishes NONE from EVENT-LABELLED records. It does "
        "not classify apnea subtypes and cannot diagnose sleep apnea."
    )

    if model_bundle is None:
        st.error("The saved model could not be loaded.")
        if model_error:
            st.code(model_error)
        st.stop()

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        st.error("The saved model bundle has an unexpected format.")
        st.stop()

    model = model_bundle["model"]
    feature_columns = model_bundle.get("feature_columns", [])

    if not feature_columns:
        st.error("The model bundle has no feature_columns list.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Model type", type(model).__name__)
    c2.metric("Input features", f"{len(feature_columns)}")
    c3.metric("Decision threshold", f"{model_threshold:.2f}")

    st.caption(
        "The decision threshold converts the model probability into a binary "
        "class. It is not a clinical threshold."
    )

    missing_features = [
        column for column in feature_columns if column not in ml_data.columns
    ]
    if missing_features:
        st.error(
            "The ML dataset is missing model features: "
            + ", ".join(missing_features)
        )
        st.stop()

    feature_data = ml_data[feature_columns].copy()
    for column in feature_columns:
        feature_data[column] = pd.to_numeric(
            feature_data[column], errors="coerce"
        )

    valid_feature_data = feature_data.loc[
        feature_data.notna().all(axis=1)
    ]

    if valid_feature_data.empty:
        st.error("No complete numeric feature rows are available.")
        st.stop()

    st.markdown("### Generate an experimental prediction")
    st.write(
        "Choose a row from the processed feature dataset. The app uses the "
        "saved model's expected features without inventing sensor values."
    )

    row_indices = valid_feature_data.index.tolist()
    selected_row_index = st.selectbox(
        "Select a feature row",
        row_indices,
        format_func=lambda idx: (
            f"Dataset row {idx}"
            + (
                f" | {ml_data.loc[idx, 'event']}"
                if "event" in ml_data.columns else ""
            )
        ),
    )

    input_row = valid_feature_data.loc[
        [selected_row_index], feature_columns
    ]

    with st.expander("View model input features"):
        st.dataframe(
            input_row.T.rename(columns={selected_row_index: "Value"}),
            use_container_width=True,
        )

    try:
        probabilities = model.predict_proba(input_row)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1

        event_probability = float(probabilities[0][positive_index])
        predicted_class = int(event_probability >= model_threshold)

        # Professional prediction card
        result_label = (
            "EVENT-LABELLED" if predicted_class == 1 else "NONE"
        )
        result_color = (
            "#FBBF24" if predicted_class == 1 else "#2DD4BF"
        )

        st.markdown("### Experimental model result")
        st.markdown(
            f"""
            <div style="
                padding: 24px;
                border-radius: 16px;
                border: 1px solid #334155;
                background: #121D2E;
                margin: 10px 0 16px 0;
            ">
                <div style="
                    font-size: 12px;
                    color: #94A3B8;
                    letter-spacing: 1px;
                ">
                    PREDICTED CLASS FOR SELECTED RECORD
                </div>
                <div style="
                    font-size: 30px;
                    font-weight: 700;
                    color: {result_color};
                    margin: 8px 0;
                ">
                    {result_label}
                </div>
                <div style="font-size: 16px; color: #CBD5E1;">
                    Experimental event-class probability:
                    <b>{event_probability:.1%}</b>
                </div>
                <div style="
                    font-size: 14px;
                    color: #94A3B8;
                    margin-top: 8px;
                ">
                    Classification threshold: {model_threshold:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            min(max(event_probability, 0.0), 1.0)
        )

        if predicted_class == 1:
            st.info(
                "The model classified this record as EVENT-LABELLED because "
                "its event-class probability meets or exceeds the selected "
                "classification threshold."
            )
        else:
            st.info(
                "The model classified this record as NONE because its "
                "event-class probability is below the classification threshold."
            )

        st.caption(
            "This is a record-level model output, not a patient-level diagnosis. "
            "NONE does not establish that sleep apnea is absent, and "
            "EVENT-LABELLED does not confirm a diagnosis."
        )

    except Exception as exc:
        st.error("The model could not generate a prediction.")
        st.exception(exc)
        event_probability = None
        predicted_class = None

    st.divider()
    st.markdown("### Global feature importance")

    if importance is not None and not importance.empty:
        feature_col = next(
            (
                column for column in [
                    "feature", "Feature", "feature_name"
                ] if column in importance.columns
            ),
            importance.columns[0],
        )
        value_col = next(
            (
                column for column in [
                    "mean_abs_shap",
                    "mean_abs_shap_value",
                    "importance",
                    "mean_abs_value",
                ] if column in importance.columns
            ),
            importance.columns[1] if len(importance.columns) > 1 else None,
        )

        if value_col:
            imp = importance[[feature_col, value_col]].copy()
            imp.columns = ["Feature", "Importance"]
            imp["Importance"] = pd.to_numeric(
                imp["Importance"], errors="coerce"
            )
            imp = imp.dropna().sort_values("Importance")

            if not imp.empty:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.barh(imp["Feature"], imp["Importance"])
                ax.set_xlabel("Saved importance value")
                ax.set_title("Global feature importance")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    imp.sort_values("Importance", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.info("The saved feature-importance CSV is unavailable.")

    st.caption(
        "Global importance summarizes which features influenced model outputs "
        "across the analyzed examples. It does not demonstrate medical causation."
    )

    st.divider()
    st.markdown("### Record-level SHAP explanation")

    if predicted_class is not None:
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_row)

            if isinstance(shap_values, list):
                shap_row = np.asarray(shap_values[-1])[0]
            else:
                shap_values = np.asarray(shap_values)
                if shap_values.ndim == 3:
                    shap_row = shap_values[0, :, -1]
                elif shap_values.ndim == 2:
                    shap_row = shap_values[0]
                else:
                    shap_row = None

            if (
                shap_row is not None
                and len(shap_row) == len(feature_columns)
            ):
                shap_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "SHAP contribution": shap_row,
                    "Feature value": input_row.iloc[0].to_numpy(),
                })
                shap_df["Absolute contribution"] = shap_df[
                    "SHAP contribution"
                ].abs()
                shap_df = shap_df.sort_values(
                    "Absolute contribution", ascending=False
                )

                plot_df = shap_df.sort_values("SHAP contribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(
                    plot_df["Feature"],
                    plot_df["SHAP contribution"],
                )
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP contribution on model output scale")
                ax.set_title("Feature contributions for selected record")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(
                    shap_df.drop(
                        columns=["Absolute contribution"]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "SHAP values describe how features contribute to the "
                    "model's output relative to its baseline. They do not "
                    "establish medical causation."
                )
            else:
                st.info(
                    "The installed SHAP version returned an unsupported output "
                    "shape for this model."
                )

        except ImportError:
            st.info("SHAP is not installed in this environment.")
        except Exception as exc:
            st.info(
                "A record-level SHAP explanation could not be generated."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


# ============================================================
# 5. EVALUATION & REPORTS
# ============================================================
elif page == "Evaluation & Reports":
    st.subheader("Evaluation & Reports")
    st.write(
        "Review the saved held-out predictions, calculate metrics from "
        "those predictions, and download supporting research outputs."
    )

    if predictions is None or predictions.empty:
        st.warning(
            "The saved test predictions are unavailable. Expected file: "
            "results/xgboost_weighted_test_predictions.csv"
        )
    elif not {"actual", "predicted"}.issubset(predictions.columns):
        st.error(
            "The prediction file must contain 'actual' and 'predicted' columns."
        )
    else:
        try:
            from sklearn.metrics import (
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            actual = pd.to_numeric(
                predictions["actual"], errors="coerce"
            )
            predicted = pd.to_numeric(
                predictions["predicted"], errors="coerce"
            )
            valid = actual.notna() & predicted.notna()

            y_true = actual[valid].astype(int)
            y_pred = predicted[valid].astype(int)

            if y_true.empty:
                st.error("No valid test predictions were found.")
            else:
                accuracy = accuracy_score(y_true, y_pred)
                balanced = balanced_accuracy_score(y_true, y_pred)
                precision = precision_score(
                    y_true, y_pred, zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, zero_division=0
                )

                auc = None
                if (
                    "event_probability" in predictions.columns
                    and y_true.nunique() == 2
                ):
                    probability = pd.to_numeric(
                        predictions.loc[valid, "event_probability"],
                        errors="coerce",
                    )
                    if probability.notna().all():
                        try:
                            auc = roc_auc_score(y_true, probability)
                        except ValueError:
                            pass

                st.markdown("### Held-out test metrics")
                a, b, c = st.columns(3)
                a.metric("Accuracy", f"{accuracy:.3f}")
                b.metric("Balanced accuracy", f"{balanced:.3f}")
                c.metric(
                    "ROC-AUC",
                    format_number(auc, digits=3)
                    if auc is not None else "N/A",
                )

                d, e, f = st.columns(3)
                d.metric("Event-class precision", f"{precision:.3f}")
                e.metric("Event-class recall", f"{recall:.3f}")
                f.metric("Event-class F1-score", f"{f1:.3f}")

                st.markdown("### Confusion matrix")
                cm = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                )

                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                image = ax.imshow(cm)
                ax.set_xticks(
                    [0, 1],
                    labels=["Predicted NONE", "Predicted EVENT"],
                )
                ax.set_yticks(
                    [0, 1],
                    labels=["Actual NONE", "Actual EVENT"],
                )
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("Actual label")
                ax.set_title("Held-out test confusion matrix")

                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j, i, str(cm[i, j]),
                            ha="center", va="center",
                        )

                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=["NONE", "EVENT-LABELLED"],
                    output_dict=True,
                    zero_division=0,
                )
                st.markdown("### Classification report")
                st.dataframe(
                    pd.DataFrame(report).T.round(4),
                    use_container_width=True,
                )

                metrics = {
                    "test_records": int(len(y_true)),
                    "accuracy": float(accuracy),
                    "balanced_accuracy": float(balanced),
                    "precision_event": float(precision),
                    "recall_event": float(recall),
                    "f1_event": float(f1),
                    "roc_auc": float(auc) if auc is not None else None,
                    "evaluation_note": (
                        "Random record-level split from a one-patient "
                        "demonstration dataset; not patient-independent validation."
                    ),
                }

                st.download_button(
                    "Download evaluation metrics (JSON)",
                    data=json.dumps(metrics, indent=2).encode("utf-8"),
                    file_name="sleepsense_evaluation_metrics.json",
                    mime="application/json",
                )

        except Exception as exc:
            st.error("Could not calculate evaluation metrics.")
            st.exception(exc)

    if predictions is not None:
        st.download_button(
            "Download saved test predictions (CSV)",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="xgboost_test_predictions.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("### Saved research outputs")

    if file_exists(CONFUSION_PATH):
        st.image(
            str(CONFUSION_PATH),
            caption="Saved XGBoost weighted-model confusion matrix",
            use_container_width=True,
        )

    if importance is not None:
        st.markdown("**Saved feature importance table**")
        st.dataframe(
            importance, hide_index=True, use_container_width=True
        )
        st.download_button(
            "Download feature importance (CSV)",
            data=importance.to_csv(index=False).encode("utf-8"),
            file_name="shap_feature_importance.csv",
            mime="text/csv",
        )

    if threshold_table is not None:
        st.markdown("**Threshold comparison results**")
        st.dataframe(
            threshold_table,
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download threshold comparison (CSV)",
            data=threshold_table.to_csv(index=False).encode("utf-8"),
            file_name="xgboost_threshold_comparison.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("### How to interpret these metrics")
    st.markdown(
        """
        - **Accuracy:** fraction of test records classified correctly.
        - **Balanced accuracy:** average recall across the two classes.
        - **Precision:** fraction of event-labelled predictions that match
          event-labelled test annotations.
        - **Recall:** fraction of event-labelled test records identified.
        - **F1-score:** harmonic mean of precision and recall.
        - **ROC-AUC:** ranking discrimination based on model probabilities.
        - **SHAP:** explanation of model behavior, not medical causation.
        """
    )

    st.error(
        "Evaluation limitation: the saved test split is random at the record "
        "level and comes from one patient. It does not demonstrate generalization "
        "to unseen patients or clinical effectiveness."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <p class="small-muted">
    SleepSense AI • Multi-Modal Sleep Apnea Detection Using PPG and SpO₂
    Signals with Explainable AI
    </p>
    <p class="small-muted">
    Educational research prototype. Not a medical device, diagnostic tool,
    or substitute for professional medical assessment.
    </p>
    """,
    unsafe_allow_html=True,
)
