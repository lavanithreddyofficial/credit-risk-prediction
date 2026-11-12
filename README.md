# Credit Risk Prediction Using Machine Learning: An Intelligent Agent Approach

## 📌 Project Overview
This project implements a machine learning classification pipeline framed as an **Intelligent Agent** to evaluate credit risk. The goal is to classify credit applicants as either **Good** (likely to repay) or **Bad** (high risk of default) based on historical financial and demographic attributes.

### Intelligent Agent Framework
* **Agent:** The decision-making entity powered by supervised machine learning classifiers (*Logistic Regression* and *Decision Tree*).
* **Environment:** The [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) dataset, consisting of 1,000 applicant records and 24 numerical attributes.
* **Objectives:** Maximize predictive performance across imbalanced classes using **Accuracy**, **F1-Score**, **Precision**, **Recall**, and **ROC-AUC**.

---

## 📊 Dataset Summary
* **Source:** UCI Machine Learning Repository (`german.data-numeric`)
* **Instances:** 1,000 applicants (0 missing values, 0 duplicates)
* **Features:** 24 numerical attributes (`Attribute_1` through `Attribute_24`)
* **Target Variable:** Binary classification (`0` = Good Credit [700 samples], `1` = Bad Credit [300 samples])

---

## 🛠️ Methodology & Pipeline
1. **Data Cleaning & Preprocessing:** 
   * Assigned standardized feature names and mapped the target variable to binary (`0` and `1`).
   * Split the dataset into **80% Training** and **20% Testing** sets (`random_state=42`).
   * Applied `StandardScaler` to normalize feature distributions.
2. **Model Building & Hyperparameter Tuning:**
   * **Logistic Regression Agent:** Tuned using `GridSearchCV` (5-fold cross-validation optimizing for F1-score). Best parameters: `{'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}`.
   * **Decision Tree Agent:** Tuned using `GridSearchCV` (5-fold cross-validation). Best parameters: `{'criterion': 'entropy', 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 5}`.
3. **Evaluation:** Compared models using Confusion Matrices, Classification Reports, and ROC Curves.

---

## 📈 Results & Comparison

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **76.50%** | **0.5524** | **0.6304** | **0.4915** | **0.7962** |
| **Decision Tree** | 69.50% | 0.4505 | 0.4808 | 0.4237 | 0.6652 |

* **Key Takeaway:** The **Logistic Regression Agent** outperformed the Decision Tree across every evaluation metric, achieving an ROC-AUC of **0.7962** and better generalization on unseen test data.

---
