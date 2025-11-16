#!/usr/bin/env python3
# Week 8 – Data Poisoning Experiments for IRIS

import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# ---------------------------------------
# CONFIG
# ---------------------------------------
POISON_RATES = [0.05, 0.10, 0.50]
POISON_TYPE = "label_flip"
WEEK_NAME = "21f3001320 Week 8: IRIS Poisoning"

# Start MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8100")
mlflow.set_experiment(WEEK_NAME)

print("📌 MLflow Tracking:", mlflow.get_tracking_uri())

# ---------------------------------------
# LOAD Iris.csv (ANY format allowed)
# ---------------------------------------
df = pd.read_csv("Iris.csv")
print("✔ Data loaded:", df.shape)

# ---------------------------------------
# AUTO-FIX COLUMNS
# ---------------------------------------
df.columns = df.columns.str.lower().str.replace(" ", "").str.replace(".", "")

rename_map = {
    "sepallengthcm": "sepal_length",
    "sepalwidthcm": "sepal_width",
    "petallengthcm": "petal_length",
    "petalwidthcm": "petal_width",
    "species": "species",
    "class": "species",
    "label": "species",
    "target": "species"
}

df = df.rename(columns=rename_map)

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# FINAL column validation
if "species" not in df.columns:
    raise ValueError("❌ Could not find species column in Iris.csv")

for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

print("✔ Column names standardized:", df.columns.tolist())

# ---------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------
train, test = train_test_split(df, test_size=0.4, stratify=df['species'], random_state=42)

X_train = train[feature_cols]
y_train = train['species']

X_test = test[feature_cols]
y_test = test['species']

# ---------------------------------------
# POISONING FUNCTIONS
# ---------------------------------------

def poison_label_flip(df, rate):
    df = df.copy()
    k = int(len(df) * rate)
    idx = np.random.choice(df.index, k, replace=False)

    unique_labels = df["species"].unique()
    for i in idx:
        true = df.loc[i, "species"]
        wrong_choices = [x for x in unique_labels if x != true]
        df.loc[i, "species"] = np.random.choice(wrong_choices)

    return df

def poison_feature_noise(df, rate, sigma=3):
    df = df.copy()
    k = int(len(df) * rate)
    idx = np.random.choice(df.index, k, replace=False)

    for col in feature_cols:
        noise = np.random.normal(0, df[col].std() * sigma, k)
        df.loc[idx, col] += noise

    return df

# ---------------------------------------
# TRAIN + LOG FUNCTION
# ---------------------------------------

def train_and_log(Xp, yp, poison_type, rate):
    params = {
        "max_depth": 2,
        "random_state": 42,
        "poison_type": poison_type,
        "poison_rate": rate
    }

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(Xp, yp)

    preds = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)

    with mlflow.start_run(run_name=f"{poison_type}_{rate}"):

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("week", "8")

        signature = infer_signature(Xp, model.predict(Xp))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="IRIS-poisoning-week8"
        )

    print(f"🔥 {poison_type} at {rate*100}% → accuracy {acc:.3f}")

# ---------------------------------------
# BASELINE RUN
# ---------------------------------------
print("\n🚀 Running baseline clean model")
train_and_log(X_train, y_train, "clean", 0.0)

# ---------------------------------------
# POISONING RUNS
# ---------------------------------------
for rate in POISON_RATES:
    print(f"\n⚠ Running poison experiment at {rate*100}%")

    if POISON_TYPE == "label_flip":
        poisoned = poison_label_flip(train, rate)
    else:
        poisoned = poison_feature_noise(train, rate)

    Xp = poisoned[feature_cols]
    yp = poisoned["species"]

    train_and_log(Xp, yp, POISON_TYPE, rate)

print("\n🎉 Week 8 Poisoning Experiments Completed Successfully!")

