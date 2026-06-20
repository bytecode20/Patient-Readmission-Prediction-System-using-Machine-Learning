# 🏥 Patient Readmission Prediction System using Machine Learning

## 📖 Overview

Hospital readmissions are one of the most significant challenges faced by healthcare organizations worldwide. Unplanned readmissions not only increase healthcare costs but may also indicate gaps in patient care and discharge planning.

This project leverages Machine Learning techniques to predict whether a patient is likely to be readmitted to the hospital within 30 days of discharge. By analyzing patient demographics, clinical history, medication information, laboratory procedures, and hospitalization patterns, the model provides a data-driven risk assessment that can help healthcare providers identify high-risk patients and take proactive interventions.

The project is built using Python, Scikit-Learn, XGBoost, and healthcare datasets from the UCI Machine Learning Repository.

---

## 🎯 Business Problem

Hospital readmissions are expensive and often preventable. Healthcare institutions strive to identify patients at risk of readmission so that additional care plans, follow-ups, and monitoring can be implemented.

This project addresses the following question:

> Can machine learning accurately predict whether a patient will be readmitted within 30 days of discharge using historical clinical data?

---

## 📊 Dataset

### Diabetes 130-US Hospitals Dataset (1999–2008)

The dataset contains over 100,000 patient encounters collected from 130 hospitals across the United States.

### Key Features

* Patient Age Group
* Gender
* Race
* Admission Type
* Time in Hospital
* Number of Medications
* Laboratory Procedures
* Number of Diagnoses
* Emergency Visits
* Inpatient Visits
* Outpatient Visits
* Diabetes Medication Information
* Clinical Diagnosis Codes

### Target Variable

The original dataset contains three classes:

| Value | Meaning                   |
| ----- | ------------------------- |
| NO    | No readmission            |
| >30   | Readmitted after 30 days  |
| <30   | Readmitted within 30 days |

For this project, the problem was transformed into binary classification:

* **1** → Readmitted within 30 days
* **0** → Not readmitted within 30 days

---

## 🏗️ Project Architecture

Patient Data
↓
Data Cleaning & Preprocessing
↓
Feature Engineering
↓
Train/Test Split
↓
XGBoost Classification Model
↓
Prediction & Risk Scoring
↓
Clinical Readmission Assessment

---

## ⚙️ Technologies Used

### Programming

* Python 3.x

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Model Serialization

* Joblib

### Data Source

* UCI Machine Learning Repository
* ucimlrepo

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

### Missing Value Handling

* Removal of highly sparse features
* Imputation of missing categorical values
* Imputation of missing numerical values

### Feature Encoding

Categorical variables such as:

* Age
* Gender
* Race
* Medication Status

were converted using One-Hot Encoding.

### Feature Selection

Relevant healthcare attributes were selected to improve model performance and reduce noise.

---

## 🤖 Machine Learning Model

### XGBoost Classifier

XGBoost was selected due to:

* High predictive performance
* Robust handling of tabular healthcare data
* Ability to capture complex nonlinear relationships
* Strong generalization capabilities

### Hyperparameters

* n_estimators = 200
* max_depth = 5
* learning_rate = 0.05
* subsample = 0.8
* colsample_bytree = 0.8

---

## 📈 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Special emphasis was placed on **Recall**, as missing a high-risk patient is more critical than generating a false positive prediction.

---

## 🚀 Prediction Workflow

1. Load trained model
2. Input patient clinical information
3. Perform preprocessing
4. Generate prediction
5. Calculate readmission probability
6. Return risk assessment

Example Output:

Readmission Risk: 72.8%

HIGH RISK: Patient likely to be readmitted within 30 days

---

## 📂 Project Structure

patient-readmission-prediction/

├── train.py

├── predict.py

├── readmission_model.pkl

├── requirements.txt

├── notebooks/

│ └── exploratory_analysis.ipynb

├── data/

│ └── dataset

└── README.md

---

## 💡 Future Enhancements

### Explainable AI

* SHAP Explainability
* Feature Importance Analysis
* Clinical Decision Transparency

### Healthcare Interoperability

* HL7 Integration
* FHIR Resource Support
* EHR Data Ingestion

### Deployment

* Streamlit Dashboard
* FastAPI Backend
* Docker Containerization
* Cloud Deployment (AWS/GCP/Azure)

### Advanced Models

* LightGBM
* CatBoost
* Deep Neural Networks
* Ensemble Learning

---

## 🌟 Key Learnings

Through this project, I gained hands-on experience in:

* Healthcare Analytics
* Machine Learning Model Development
* Clinical Data Processing
* Feature Engineering
* Model Evaluation
* Predictive Healthcare Applications

This project demonstrates how machine learning can support data-driven healthcare decisions and contribute to improving patient outcomes while reducing hospital readmission rates.

---

## 👨‍💻 Author

**Arjun**

Software Engineer | Healthcare Integration Specialist | Aspiring AI & Data Engineer

### Skills Demonstrated

✔ Machine Learning

✔ Healthcare Analytics

✔ Python Development

✔ Data Preprocessing

✔ Feature Engineering

✔ Predictive Modeling

✔ Healthcare Technology

✔ Clinical Data Analysis
