from extractor import extract_symptoms
from predict import predict
from treatment_lookup import get_treatment
from chatbot import generate_response


def process_symptoms(user_input):

    # -----------------------------
    # STEP 1: EXTRACT
    # -----------------------------
    extracted = extract_symptoms(user_input)

    if len(extracted) < 2:
        return "🤖 Please provide at least 2–3 symptoms for better analysis."

    # -----------------------------
    # STEP 2: PREDICT (MLP)
    # -----------------------------
    results = predict(extracted)

    if isinstance(results, str):
        return results

    if not results:
        return "🤖 Unable to determine condition. Please try again."

    # -----------------------------
    # STEP 3: TAKE TOP 2
    # -----------------------------
    top_diseases = results[:2]

    # -----------------------------
    # STEP 4: BUILD TEXT FOR CHATBOT
    # -----------------------------
    final_text = ""

    for disease, prob in top_diseases:

        treatment = get_treatment(disease)

        if treatment:
            med1 = treatment.get("med1", "N/A")
            med2 = treatment.get("med2", "N/A")
            advice = treatment.get("advice", "No advice available")

            if str(med2) == "nan":
                med2 = "N/A"
        else:
            med1 = "Consult doctor"
            med2 = "N/A"
            advice = "Seek medical attention if symptoms persist."

        final_text += f"""
Disease: {disease} ({prob}%)
Medicine1: {med1}
Medicine2: {med2}
Advice: {advice}
"""

    # -----------------------------
    # STEP 5: CHATBOT RESPONSE
    # -----------------------------
    return generate_response(user_input, final_text)