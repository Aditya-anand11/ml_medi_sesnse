import json
import pandas as pd
from joblib import load
import os

# -----------------------------
# LOAD FEATURES
# -----------------------------
with open("saved_models/feature_columns.json") as f:
    FEATURES = json.load(f)

# -----------------------------
# LOAD LABEL ENCODER (OPTIONAL)
# -----------------------------
try:
    le = load("dataset/label_encoder.pkl")
except:
    le = None

# -----------------------------
# LOAD MLP MODEL
# -----------------------------
MODEL_PATH = "saved_models/mlp.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ MLP model not found")

model = load(MODEL_PATH)

# -----------------------------
# CREATE INPUT VECTOR
# -----------------------------
def create_input_vector(symptoms):
    row = dict.fromkeys(FEATURES, 0)

    for s in symptoms:
        if s in row:
            row[s] = 1

    return pd.DataFrame([row])[FEATURES]


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(symptoms, top_k=5):

    df = create_input_vector(symptoms)

    # -----------------------------
    # GET PROBABILITIES
    # -----------------------------
    probs = model.predict_proba(df)[0]
    classes = model.classes_

    results = []

    for i in range(len(classes)):
        prob = round(probs[i] * 100, 2)

        # convert label → disease name
        if le:
            disease = le.inverse_transform([classes[i]])[0]
        else:
            disease = str(classes[i])

        results.append((disease, prob))

    # -----------------------------
    # SORT + TOP K
    # -----------------------------
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results[:top_k]