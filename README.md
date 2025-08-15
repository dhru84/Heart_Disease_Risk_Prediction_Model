# Cardiovascular Risk Prediction

This project predicts the **risk of heart disease** using machine learning techniques.
It uses patient health indicators such as **age, cholesterol, blood pressure, BMI, glucose levels**, and more to determine the likelihood of cardiovascular disease.

The model is implemented in Python using **Pandas**, **NumPy**, and **Scikit-learn** with **visualizations in Matplotlib/Seaborn and optional interactive inputs via ipywidgets**.

---

## Project Overview

* **Goal:** Predict the probability of heart disease risk based on clinical parameters.
* **Dataset:** `cardiovascular_risk.csv`
  The dataset includes multiple medical and lifestyle-related features.
* **Machine Learning Model:** Logistic Regression, Random Forest Classifier
* **Evaluation Metrics:** Accuracy, ROC-AUC, classification report, confusion matrix; model interpretability via coefficients (LogReg) and feature importances (Random Forest)

---

## Workflow

1. **Data Loading & Exploration**

   * Load dataset from CSV, inspect shape and columns, preview samples.
   * Basic EDA and type checks.
2. **Data Preprocessing**

   * Handle missing or inconsistent values.
   * Train/validation splits with stratification.
   * Standardize features using `StandardScaler`.
3. **Model Training**

   * Train a **Logistic Regression** model on the training set.
4. **Model Evaluation**

   * Baseline Logistic Regression.
   * Random Forest Classifier with tuned n_estimators and fixed random_state for reproducibility.
   * Cross-validation using StratifiedKFold.
5. **Prediction**

   * Predict heart disease risk for unseen data.

---

## Installation & Setup

To run the notebook locally, ensure you have the following installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn ipywidgets
```

---

## Model Performance

**Reports include:**

  * Accuracy and ROC-AUC on validation splits.
  * Classification report (precision, recall, F1-score).
  * Confusion matrix heatmap.
  * ROC curves for visual comparison of models.

**Feature interpretation:**

  * Logistic Regression: coefficient magnitudes and signs.
  * Random Forest: feature importance rankings.

**Actual metric values will be printed in the notebook after training/evaluation cells are executed.

---

## Project Structure

* Heart_Disease_Risk_Prediction_Model.ipynb — notebook with complete pipeline.
* cardiovascular_risk.csv — dataset used for training and evaluation.

---

## Key Skills Demonstrated

* Data Cleaning & Preprocessing
* Stratified train/validation splits and cross-validation
* Feature scaling and ML pipelines
* Binary classification with Logistic Regression and Random Forest
* Model evaluation (Accuracy, ROC-AUC, Precision, Recall, F1)
* Visualization (confusion matrix, ROC)
* Interactive prediction UI with ipywidgets

---

## License

This project is released under the MIT License.
