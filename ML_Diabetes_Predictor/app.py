from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features or len(features) != 8:
        return jsonify({"error": "Invalid input"}), 400

    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]  

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
