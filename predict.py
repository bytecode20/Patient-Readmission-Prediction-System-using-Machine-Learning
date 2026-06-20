import joblib
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------

# LOAD TRAINED MODEL

# ---------------------------------------------------

model = joblib.load("readmission_model.pkl")

# ---------------------------------------------------

# LOAD DATASET

# ---------------------------------------------------

dataset = fetch_ucirepo(id=296)

X = dataset.data.features

# ---------------------------------------------------

# APPLY SAME COLUMN DROPS AS TRAINING

# ---------------------------------------------------

columns_to_drop = [
"weight",
"payer_code",
"medical_specialty"
]

X = X.drop(
columns=columns_to_drop,
errors="ignore"
)

# ---------------------------------------------------

# SELECT A SAMPLE PATIENT

# ---------------------------------------------------

sample_patient = X.iloc[[0]]

print("Patient Data:")
print(sample_patient)

# ---------------------------------------------------

# PREDICT

# ---------------------------------------------------

prediction = model.predict(sample_patient)

probability = model.predict_proba(sample_patient)[0][1]

# ---------------------------------------------------

# RESULTS

# ---------------------------------------------------

print("\nPrediction Result")
print(f"Readmission Risk: {probability:.2%}")

if prediction[0] == 1:
    print("HIGH RISK: Patient likely to be readmitted within 30 days")
else:
    print("LOW RISK: Patient unlikely to be readmitted within 30 days")
