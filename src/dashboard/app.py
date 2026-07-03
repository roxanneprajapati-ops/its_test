"""
app.py (Dashboard)
-------------------
Streamlit dashboard - the "Application/Visualization Layer" of the
ITS architecture. Five page matching the rubric deliverable:

    1. Executive Dashboard       - headline KPI
    2. Traffic Trends             - per-intersection time series
    3. ML Model Results            - compare all trained model
    4. Signal Recommendations       - live adaptive signal demo
    5. Security & Audit             - simulated attack test result

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json

import pandas as pd
import streamlit as st

from src.models.classification_models import CLASSIFICATION_FEATURES, RandomForestCongestionClassifier
from src.optimization.optimizer import SignalOptimizationEngine
from src.utils import config

st.set_page_config(page_title="Auckland ITS Traffic Dashboard", layout="wide")


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DATASET_PATH, parse_dates=["Timestamp"])


@st.cache_data
def load_metrics() -> dict:
    path = config.OUTPUT_DIR / "metrics.json"

    if not path.exists():
        st.warning(f"metrics.json not found at: {path}")
        return {}

    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_comparison() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "baseline_vs_adaptive.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_security_results() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "security_test_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_classifier() -> RandomForestCongestionClassifier:
    """Load trained congestion classifier."""
    model_path = config.MODELS_DIR / "congestion_random_forest_classifier.joblib"

    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    clf = RandomForestCongestionClassifier()
    clf.load()
    return clf


def page_executive_dashboard(df: pd.DataFrame, comparison: pd.DataFrame):
    st.header("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Speed (km/h)", f"{df['Avg_Speed'].mean():.1f}")
    col2.metric("Average Queue Length", f"{df['Queue_Length'].mean():.1f}")
    high_pct = (df["Congestion_Level"] == "High").mean() * 100
    col3.metric("High Congestion Rate", f"{high_pct:.1f}%")
    col4.metric("Total Incidents (simulated)", int(df["Incident_Flag"].sum()))

    st.subheader("Congestion Level Distribution")
    st.bar_chart(df["Congestion_Level"].value_counts())

    if not comparison.empty:
        st.subheader("Fixed-Time vs AI-Adaptive Signal: Delay Reduction")
        st.bar_chart(comparison.set_index("Intersection_ID")["Delay_Reduction_Pct"])


def page_traffic_trends(df: pd.DataFrame):
    st.header("Traffic Trends")
    intersection = st.selectbox("Select Intersection", config.INTERSECTION_IDS)
    subset = df[df["Intersection_ID"] == intersection].sort_values("Timestamp")

    st.subheader(f"Vehicle Count - {config.INTERSECTION_NAMES.get(intersection, intersection)}")
    st.line_chart(subset.set_index("Timestamp")["Vehicle_Count"])

    st.subheader("Average Speed")
    st.line_chart(subset.set_index("Timestamp")["Avg_Speed"])

    st.subheader("Queue Length")
    st.line_chart(subset.set_index("Timestamp")["Queue_Length"])


def page_ml_results(metrics: dict):
    st.header("Machine Learning Results")
    if not metrics:
        st.warning("No metrics.json found. Run `python main.py train` first.")
        return

    st.subheader("Congestion Classifiers")
    for name, m in metrics.get("classification", {}).items():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{name} Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision_weighted']:.2%}")
        c3.metric("Recall", f"{m['recall_weighted']:.2%}")
        c4.metric("F1", f"{m['f1_weighted']:.2%}")

    st.subheader("Queue Length Regressors")
    for name, m in metrics.get("regression", {}).items():
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{name} MAE", f"{m['mae']:.2f}")
        c2.metric("RMSE", f"{m['rmse']:.2f}")
        c3.metric("R\u00b2", f"{m['r2']:.3f}")

    st.subheader("Incident Detectors (recall-tuned)")
    for name, m in metrics.get("incident_detection", {}).items():
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{name} Recall", f"{m['tuned_recall']:.2%}")
        c2.metric("Precision", f"{m['tuned_precision']:.2%}")
        c3.metric("F1", f"{m['tuned_f1']:.2%}")
    st.caption(metrics.get("incident_tradeoff_explanation", ""))


def page_signal_recommendations(df: pd.DataFrame, classifier: RandomForestCongestionClassifier):
    st.header("Live Signal Recommendations")
    optimizer_engine = SignalOptimizationEngine()
    latest = df.sort_values("Timestamp").groupby("Intersection_ID").tail(1).reset_index(drop=True)

    for _, row in latest.iterrows():
        features = pd.DataFrame([row[CLASSIFICATION_FEATURES]])
        predicted_level = classifier.predict(features)[0]
        recommendation = optimizer_engine.recommend(
            row["Intersection_ID"], predicted_level,
            queue_length=int(row["Queue_Length"]),
            incident_detected=bool(row["Incident_Flag"]),
        )
        name = config.INTERSECTION_NAMES.get(row["Intersection_ID"], row["Intersection_ID"])
        with st.container(border=True):
            st.subheader(name)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Congestion", predicted_level)
            c2.metric("Fixed-Time Green", f"{recommendation.fixed_time_green}s")
            c3.metric("AI-Adaptive Green", f"{recommendation.recommended_green_time}s")
            st.caption(recommendation.reason)


def page_security(security_df: pd.DataFrame):
    st.header("Security & Audit")
    if security_df.empty:
        st.warning("No security_test_results.csv found. Run `python main.py train` first.")
    else:
        st.subheader("Simulated Attack Test Results")
        st.dataframe(security_df)
        passed = int(security_df["passed"].sum())
        st.metric("Attacks Correctly Blocked", f"{passed}/{len(security_df)}")

    if config.AUDIT_LOG_PATH.exists():
        st.subheader("Recent Audit Log")
        st.dataframe(pd.read_csv(config.AUDIT_LOG_PATH).tail(20))


def main():
    st.title("Auckland AI-Based Adaptive Traffic Signal Optimization Dashboard")
    df = load_processed_data()
    metrics = load_metrics()
    comparison = load_comparison()
    security_df = load_security_results()

    page = st.sidebar.radio(
        "Navigate",
        ["Executive Dashboard", "Traffic Trends", "ML Model Results",
         "Signal Recommendations", "Security & Audit"],
    )

    if page == "Executive Dashboard":
        page_executive_dashboard(df, comparison)
    elif page == "Traffic Trends":
        page_traffic_trends(df)
    elif page == "ML Model Results":
        page_ml_results(metrics)
    elif page == "Signal Recommendations":
        classifier = load_classifier()
        page_signal_recommendations(df, classifier)
    elif page == "Security & Audit":
        page_security(security_df)


if __name__ == "__main__":
    main()
