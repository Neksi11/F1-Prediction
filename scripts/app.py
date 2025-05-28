# app.py
from flask import Flask, jsonify, request
import pandas as pd
import random
import joblib
import os

app = Flask(__name__)

# Load trained model
model_path = r"C:\Users\Neksi\Desktop\F1-winner-model\models_trained\xgb_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found: xgb_model.pkl")
model = joblib.load(model_path)

# Initialize mock driver data
drivers = ['VER', 'HAM', 'LEC', 'ALO', 'NOR', 'SAI', 'PER', 'RUS', 'BOT', 'TSU']
driver_data = pd.DataFrame({
    'driver': drivers,
    'grid': [random.randint(1, 10) for _ in drivers],
    'position_gain': [0 for _ in drivers],
    'avg_grid_last_5': [random.uniform(2, 8) for _ in drivers],
    'avg_position_last_5': [random.uniform(2, 9) for _ in drivers],
    'avg_points_last_5': [random.uniform(5, 25) for _ in drivers]
})

lap_number = 0

@app.route("/simulate-lap", methods=["GET"])
def simulate_lap():
    global lap_number, driver_data
    lap_number += 1

    # Update position gain randomly
    driver_data['position_gain'] += [random.choice([-1, 0, 1]) for _ in drivers]
    driver_data['position_gain'] = driver_data['position_gain'].clip(-5, 15)

    # Predict win probabilities
    features = ['grid', 'position_gain', 'avg_grid_last_5', 'avg_position_last_5', 'avg_points_last_5']
    X = driver_data[features]
    driver_data['win_probability'] = model.predict_proba(X)[:, 1]

    # Prepare response
    results = driver_data.sort_values(by='win_probability', ascending=False)[['driver', 'grid', 'position_gain', 'win_probability']]
    response = {
        "lap": lap_number,
        "results": results.to_dict(orient="records")
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)


   # http://127.0.0.1:5000/simulate-lap   <------ [ Go to this URL after starting the server ]
