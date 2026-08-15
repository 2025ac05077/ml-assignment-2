"""
ML Assignment 2 - Model Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: sklearn.datasets (originally from UCI ML Repository)
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Task: Binary classification (Malignant vs Benign)
Instances: 569 (>= 500 required)
Features: 30 (>= 12 required)

This script:
 1. Loads the dataset
 2. Splits into train/test
 3. Scales features
 4. Trains 5 classification models:
    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbor Classifier
    - Naive Bayes Classifier (Gaussian)
    - Random Forest Classifier (Ensemble)
 5. Evaluates each model: Accuracy, AUC, Precision, Recall, F1, MCC
 6. Saves each trained model + the scaler to disk (joblib) inside model/
 7. Saves a test_data.csv (features + true label) at project root for the Streamlit app
 8. Prints/saves a comparison table (comparison_metrics.csv)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project-folder/
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(BASE_DIR, "comparison_metrics.csv")


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    return X, y, data.target_names


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(list(target_names), os.path.join(MODEL_DIR, "target_names.joblib"))

    models = get_models()
    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics["ML Model Name"] = name
        results.append(metrics)

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"Trained and saved: {name} -> {filename}")
        print(f"  Metrics: {metrics}")

    # Comparison table
    df_results = pd.DataFrame(results)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    df_results.to_csv(METRICS_PATH, index=False)
    print("\nComparison Table:\n", df_results)

    # Save test data (raw features, unscaled) + true label for the Streamlit app
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved test data to: {TEST_DATA_PATH}")
    print(f"Saved comparison metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
