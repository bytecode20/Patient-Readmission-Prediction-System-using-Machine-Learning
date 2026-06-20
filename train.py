from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

dataset = fetch_ucirepo(id=296)

X = dataset.data.features
y = dataset.data.targets

print("Dataset Shape:", X.shape)

# ---------------------------------------------------
# TARGET PROCESSING
# ---------------------------------------------------

# Convert readmitted into binary
y = y["readmitted"].apply(
    lambda x: 1 if x == "<30" else 0
)

print("\nTarget Distribution:")
print(y.value_counts())

# ---------------------------------------------------
# DROP HIGH MISSING COLUMNS
# ---------------------------------------------------

columns_to_drop = [
    "weight",
    "payer_code",
    "medical_specialty"
]

for col in columns_to_drop:
    if col in X.columns:
        X = X.drop(columns=[col])

# ---------------------------------------------------
# IDENTIFY COLUMN TYPES
# ---------------------------------------------------

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns:", len(categorical_cols))
print("Numerical Columns:", len(numerical_cols))

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_cols
        ),
        (
            "cat",
            categorical_transformer,
            categorical_cols
        )
    ]
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining...")

pipeline.fit(
    X_train,
    y_train
)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"\nROC AUC Score: {auc:.4f}")

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    pipeline,
    "readmission_model.pkl"
)

print("\nModel Saved Successfully!")