"""
Disease Prediction - Inference App
====================================
Run: python app.py
"""

import pandas as pd
import pickle

# ─────────────────────────────────────────────────────────────
# LOAD MODEL & LABEL ENCODER
# ─────────────────────────────────────────────────────────────
print("Loading model...")
with open("saved_models/MLP.pkl", "rb") as f:
    model = pickle.load(f)
print("  Model loaded")

print("Loading label encoder...")
with open("dataset/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
print("  Label encoder loaded")

# ── Sanity check ──────────────────────────────────────────
print(f"\nEncoder check (first 5 classes):")
for i, name in enumerate(le.classes_[:5]):
    print(f"  {i} -> {name}")

print(f"\nTotal features : {len(model.feature_names_in_)}")
print(f"Total classes  : {len(le.classes_)}")
print("\nReady! Type symptoms below.")
print("─" * 50)

# ─────────────────────────────────────────────────────────────
# INPUT LOOP
# ─────────────────────────────────────────────────────────────
while True:

    user_input = input("\nDescribe your symptoms (comma separated) or 'exit': ")

    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    # ── Parse symptoms ────────────────────────────────────
    symptoms = [s.strip().lower() for s in user_input.split(",")]
    print(f"  Parsed: {symptoms}")

    # ── Build feature vector ──────────────────────────────
    row = {feature: 0 for feature in model.feature_names_in_}

    matched   = []
    unmatched = []

    for s in symptoms:
        if s in row:
            row[s] = 1
            matched.append(s)
        else:
            unmatched.append(s)

    if unmatched:
        print(f"  Warning - unknown symptoms (ignored): {unmatched}")

    if not matched:
        print("  No valid symptoms found. Please try again.")
        continue

    print(f"  Matched symptoms: {matched}")

    df_input = pd.DataFrame([row])[model.feature_names_in_]

    # ── Predict ───────────────────────────────────────────
    probs = model.predict_proba(df_input)[0]

    # Map class index -> disease name -> probability
    results = []
    for class_idx, prob in enumerate(probs):
        # model.classes_ gives the actual label indices used during training
        label = model.classes_[class_idx]
        try:
            disease = le.inverse_transform([label])[0]
        except Exception:
            disease = f"Unknown_{label}"
        results.append((disease, round(prob * 100, 2)))

    # Sort by probability descending
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # ── Output ────────────────────────────────────────────
    print("\n🩺 Top 10 Predictions:\n")
    for i, (disease, prob) in enumerate(results[:10], 1):
        bar = "█" * int(prob / 2)
        print(f"  {i:>2}. {disease:<45} {prob:>6}%  {bar}")