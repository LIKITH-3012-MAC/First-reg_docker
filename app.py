from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return {
        "message": "Marks Prediction API is running 🚀",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {
                "study_hours": "number (int or float)"
            }
        }
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "study_hours" not in data:
        return jsonify({"error": "study_hours is required"}), 400

    try:
        study_hours = float(data["study_hours"])
    except ValueError:
        return jsonify({"error": "study_hours must be a number"}), 400

    # Model expects 2D array
    prediction = model.predict(np.array([[study_hours]]))

    return jsonify({
        "study_hours": study_hours,
        "predicted_marks": round(float(prediction[0]), 2)
    })


# 🔥 REQUIRED FOR RAILWAY / CLOUD
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
