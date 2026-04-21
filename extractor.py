import ollama
import json
import difflib

# -----------------------------
# LOAD FEATURE LIST
# -----------------------------
with open("saved_models/feature_columns.json") as f:
    FEATURES = json.load(f)

# -----------------------------
# EXTRACT SYMPTOMS USING LLM
# -----------------------------
def extract_symptoms(user_input):

    prompt = f"""
Extract all symptoms from the sentence below.

Rules:
- Output ONLY comma-separated symptoms
-case sensitive
- No explanation


Sentence:
{user_input}
"""

    try:
        res = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        text = res["message"]["content"]

        # remove reasoning if present
        if "</think>" in text:
            text = text.split("</think>")[-1]

        raw = text.strip()

    except Exception as e:
        print("⚠️ LLM Error:", e)
        return []

    # -----------------------------
    # MAP TO DATASET FEATURES
    # -----------------------------
    cleaned = []

    for s in raw.split(","):
        s = s.strip().lower()

        # normalize format
        s = s.replace("-", " ").replace("_", " ")
        s = s.strip()

        # 🔥 fuzzy match to dataset features
        match = difflib.get_close_matches(
            s,
            [f.replace("_", " ") for f in FEATURES],
            n=1,
            cutoff=0.6
        )

        if match:
            # convert back to dataset format
            matched_feature = match[0].replace(" ", "_")
            cleaned.append(matched_feature)

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    return list(set(cleaned))