from extractor import extract_symptoms
from predict import predict
from treatment_lookup import get_treatment
from chatbot import generate_response


def run_pipeline(user_input):

    print("\n" + "="*60)
    print("🧠 Processing...\n")

    # -----------------------------
    # STEP 1: EXTRACT SYMPTOMS
    # -----------------------------
    extracted = extract_symptoms(user_input)
    print("🧠 Extracted Symptoms:", extracted)

    if not extracted or len(extracted) < 2:
        print("\n❌ Please provide at least 2–3 clear symptoms.\n")
        return

    # -----------------------------
    # STEP 2: PREDICT (MLP)
    # -----------------------------
    results = predict(extracted)

    if isinstance(results, str):
        print(results)
        return

    # -----------------------------
    # STEP 3: SHOW RAW PREDICTIONS
    # -----------------------------
    print("\n🩺 Top Predictions:\n")

    for i, (disease, prob) in enumerate(results, 1):
        bar = "█" * int(prob / 2)
        print(f"{i}. {disease:<40} {prob:>6}%  {bar}")

    # -----------------------------
    # STEP 4: TAKE TOP 2
    # -----------------------------
    top_2 = results[:2]

    # -----------------------------
    # STEP 5: BUILD CHATBOT INPUT
    # -----------------------------
    final_text = ""

    for disease, prob in top_2:
        treatment = get_treatment(disease)

        if treatment:
            med1 = treatment.get("med1", "N/A")
            med2 = treatment.get("med2", "N/A")
            advice = treatment.get("advice", "No advice available")

            if str(med2) == "nan":
                med2 = "N/A"

            final_text += f"""
Disease: {disease} ({prob}%)
Medicine1: {med1}
Medicine2: {med2}
Advice: {advice}
"""
        else:
            final_text += f"""
Disease: {disease} ({prob}%)
No treatment data available
"""

    # -----------------------------
    # STEP 6: CHATBOT RESPONSE
    # -----------------------------
    print("\n🤖 Chatbot Response:\n")

    response = generate_response(user_input, final_text)
    print(response)

    print("\n" + "="*60 + "\n")


# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":

    print("\n=== AI Disease Prediction System ===\n")

    while True:
        user_input = input("Describe your symptoms (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("\n👋 Exiting system. Stay healthy!\n")
            break

        run_pipeline(user_input)