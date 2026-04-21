import ollama

def generate_response(user_input, diseases_with_treatment):

    prompt = f"""
You are an intelligent medical assistant.

Patient symptoms:
{user_input}

Predicted conditions and treatments:
{diseases_with_treatment}

Instructions:
- Explain in simple human language
- Mention most likely condition first
- Briefly explain why
- Include medicines and advice clearly
- DO NOT sound robotic
- DO NOT say "AI model predicts"
- Add a short safety disclaimer
-no follow up questions, just a clear answer

Output should feel like a real doctor conversation.
"""

    try:
        res = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7}
        )

        text = res["message"]["content"]

        if "</think>" in text:
            text = text.split("</think>")[-1]

        return text.strip()

    except Exception as e:
        return f"⚠️ Chatbot error: {e}"