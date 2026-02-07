from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return jsonify({
        "message": "Regression Model API 🚀",
        "environment": os.getenv("ENVIRONMENT", "dev")
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    hours = data.get("study_hours")

    if hours is None:
        return jsonify({"error": "study_hours is required"}), 400

    prediction = model.predict(np.array([[hours]]))

    return jsonify({
        "study_hours": hours,
        "predicted_marks": round(prediction[0], 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

