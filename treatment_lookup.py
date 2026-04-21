import pandas as pd

# load once
df = pd.read_csv("dataset/treatment.csv")

def get_treatment(disease):

    # normalize
    disease = disease.strip().lower()

    row = df[df["Disease"].str.lower() == disease]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "med1": row["Medication_1"],
        "med2": row["Medication_2"],
        "advice": row["Lifestyle_Advice"]
    }