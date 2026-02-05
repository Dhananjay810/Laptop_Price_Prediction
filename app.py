from flask import Flask, request, jsonify
import pandas as pd
import pickle

# ---------------------------
# Load trained pipeline
# ---------------------------
with open("pipe.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

# 🔴 EXACT columns used during training
EXPECTED_COLUMNS = [
    'Company',
    'TypeName',
    'Inches',
    'OpSys',
    'W',
    'fwe',
    'ram',
    'CpuModel',
    'CpuParta',
    'CpuPartc',
    'CpuPartb',
    'efe'
]

@app.route("/")
def home():
    return jsonify({"message": "Laptop Price Prediction API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert JSON → DataFrame
        df = pd.DataFrame([data])

        # 🔴 Align columns exactly as training
        df = df.reindex(columns=EXPECTED_COLUMNS)

        # 🔴 Replace missing values (VERY IMPORTANT)
        df = df.fillna(0)

        # Predict
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ---------------------------
# Run locally / Render
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
