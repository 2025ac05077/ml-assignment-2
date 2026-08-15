# Breast Cancer Classification - ML Assignment 2

## a. Problem Statement

The goal of this project is to build and compare multiple machine learning
classification models that predict whether a breast tumor is **malignant**
or **benign** based on features computed from digitized images of a fine
needle aspirate (FNA) of a breast mass. This is a **binary classification**
problem. An interactive Streamlit web application is built to demonstrate
model predictions and evaluation metrics, and the app is deployed on
Streamlit Community Cloud.

## b. Dataset Description

- **Dataset name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / available directly via
  `sklearn.datasets.load_breast_cancer`
  (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Number of instances:** 569
- **Number of features:** 30 numeric features (e.g., radius, texture,
  perimeter, area, smoothness, compactness, concavity, symmetry, fractal
  dimension — each computed as mean, standard error, and "worst"/largest value)
- **Target classes:** `0 = malignant`, `1 = benign`
- **Task type:** Binary classification

## c. GitHub Repository Link

> https://github.com/2025ac05077/ml-assignment-2

## d. Models Used

The following 5 classification models were trained on the same dataset
(80/20 train-test split, features standardized using `StandardScaler`):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (k=5)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble, 200 trees)

### Comparison Table

(Generated from `comparison_metrics.csv` — 80/20 stratified train-test split, `random_state=42`)

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression         | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree                | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                          | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                  | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)     | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name              | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Best overall performer on this dataset; the classes are close to linearly separable after scaling, so a simple linear decision boundary works extremely well. Highest accuracy, F1, and MCC. |
| Decision Tree                | Weakest performer; a single tree overfits the training data and generalizes worst among all models, reflected in the lowest MCC and AUC. |
| kNN                          | Strong performance after feature scaling (critical for distance-based models); ties with Random Forest on Accuracy/F1/MCC. |
| Naive Bayes                  | Decent AUC despite the (violated) feature-independence assumption, but lower precision/recall balance than Logistic Regression or ensembles. |
| Random Forest (Ensemble)     | Very strong and stable performance — the ensemble of trees corrects the overfitting problem seen in the single Decision Tree, and it has the 2nd highest AUC overall. |
| **Overall Winner**           | **Logistic Regression** — highest Accuracy (0.9825), AUC (0.9954), F1 (0.9861), and MCC (0.9623) on this dataset. |

## e. Streamlit App

- **Live App Link:** > https://2025ac05077-ml-assignment-2.streamlit.app/
- **Features:**
  - Dataset upload option (CSV) — upload `test_data.csv`
  - Model selection dropdown (choose among the 5 trained models)
  - Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  - Confusion matrix and classification report visualization
  - "Compare All Models" button to see all 5 models' metrics side-by-side

## f. Project Structure

```
ml_assignment2/
│-- app.py                  (Streamlit app)
│-- requirements.txt
│-- README.md
│-- test_data.csv           (generated after running train_models.py)
│-- comparison_metrics.csv  (generated after running train_models.py)
│-- model/
│   │-- train_models.py     (trains and saves all 5 models + scaler)
│   │-- scaler.joblib
│   │-- feature_names.joblib
│   │-- target_names.joblib
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest_ensemble.joblib
```

## g. How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates model files, test_data.csv, comparison_metrics.csv)
python model/train_models.py

# 3. Run the Streamlit app
streamlit run app.py
```

Then upload `test_data.csv` in the app sidebar, select a model, and view results.

## h. BITS Virtual Lab Screenshot

### 1. VS Code Environment in BITS Virtual Lab
![BITS Virtual Lab - VS Code](<Screenshot 2026-08-15 at 10.52.30 AM.png>)

### 2. Streamlit Application Running in BITS Virtual Lab
![BITS Virtual Lab - Streamlit App](<Screenshot 2026-08-15 at 10.54.56 AM.png>)
