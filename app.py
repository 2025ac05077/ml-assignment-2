"""
ML Assignment 2 - Streamlit App
Interactive frontend to:
 - Upload test data (CSV)
 - Select a trained model
 - View evaluation metrics
 - View confusion matrix / classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    target_names = joblib.load(os.path.join(MODEL_DIR, "target_names.joblib"))
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return scaler, feature_names, target_names, models


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classification Demo", layout="wide")
    st.title("🔬 Breast Cancer Classification - Model Comparison App")
    st.markdown(
        """
        Upload the **test_data.csv** (or any CSV with the same feature columns + a `target` column),
        choose a model, and view its performance.
        """
    )

    scaler, feature_names, target_names, models = load_artifacts()

    st.sidebar.header("⚙️ Options")
    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
    model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

    if uploaded_file is None:
        st.info("👈 Please upload a CSV file (e.g., test_data.csv from the repo) to begin.")
        return

    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with true labels.")
        return

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        return

    X = df[feature_names]
    y_true = df["target"]

    X_scaled = scaler.transform(X)

    model = models[model_name]
    y_pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = y_pred

    st.subheader(f"📊 Results for: {model_name}")

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metric_cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(metric_cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=[str(t) for t in target_names], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📈 Compare All Models")
    if st.button("Run comparison on uploaded data for all models"):
        all_results = []
        for name, m in models.items():
            yp = m.predict(X_scaled)
            ypr = m.predict_proba(X_scaled)[:, 1] if hasattr(m, "predict_proba") else yp
            met = compute_metrics(y_true, yp, ypr)
            met["ML Model Name"] = name
            all_results.append(met)
        comp_df = pd.DataFrame(all_results)[
            ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ]
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=comp_df.columns[1:]))


if __name__ == "__main__":
    main()
