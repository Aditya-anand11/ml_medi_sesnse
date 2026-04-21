# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from main_logic import process_symptoms

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    response = process_symptoms(text)
    return jsonify({"response": response})

app.run(debug=True)