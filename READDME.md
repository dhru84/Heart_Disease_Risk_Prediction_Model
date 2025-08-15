# Cardiovascular Risk Prediction

This project predicts the **risk of heart disease** using machine learning techniques.
It uses patient health indicators such as **age, cholesterol, blood pressure, BMI, glucose levels**, and more to determine the likelihood of cardiovascular disease.

The model is implemented in Python using **Pandas**, **NumPy**, and **Scikit-learn**.

---

## Project Overview

* **Goal:** Predict the probability of heart disease risk based on clinical parameters.
* **Dataset:** `data_cardiovascular_risk.csv`
  The dataset includes multiple medical and lifestyle-related features.
* **Machine Learning Model:** Logistic Regression
* **Evaluation Metrics:** Accuracy Score, Classification Report, Confusion Matrix

---

## Workflow

1. **Data Loading & Exploration**

   * Load dataset from CSV.
   * View structure, summary statistics, and data types.
2. **Data Preprocessing**

   * Handle missing or inconsistent values.
   * Standardize features using `StandardScaler`.
3. **Model Training**

   * Train a **Logistic Regression** model on the training set.
4. **Model Evaluation**

   * Evaluate performance using **Accuracy**, **Precision**, **Recall**, and **F1-score**.
5. **Prediction**

   * Predict heart disease risk for unseen data.

---

## Installation & Setup

To run the notebook locally, ensure you have the following installed:

```bash
pip install numpy pandas scikit-learn
```

---

## Model Performance

* **Algorithm:** Logistic Regression
* **Accuracy:** **86.87%**
* **Precision (Class 0 / Class 1):** 0.87 / 0.83
* **Recall (Class 0 / Class 1):** 1.00 / 0.10
* **F1-Score (Class 0 / Class 1):** 0.93 / 0.18
* **Confusion Matrix:**

  ```
  [[579   2]
   [ 87  10]]
  ```

---

## Key Skills Demonstrated

* Data Cleaning & Preprocessing
* Feature Scaling
* Logistic Regression Modeling
* Model Evaluation (Accuracy, Precision, Recall, F1-Score)
* Confusion Matrix Analysis

---

## License

This project is released under the MIT License.
