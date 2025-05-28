# lap_simulation.py
import time
import random
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

# Load your trained XGBoost model
model = joblib.load(r"C:\Users\Neksi\Desktop\F1-winner-model\models_trained\xgb_model.pkl")

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", "lap_simulation_results.txt")

# Prepare to write results to file
with open(output_path, "w") as f_out:
    f_out.write("Lap-by-Lap F1 Race Simulation Results\n")
    f_out.write("===================================\n")

    # Mock driver grid and stats
    drivers = ['VER', 'HAM', 'LEC', 'ALO', 'NOR', 'SAI', 'PER', 'RUS', 'BOT', 'TSU']
    driver_data = pd.DataFrame({
        'driver': drivers,
        'grid': [random.randint(1, 10) for _ in drivers],
        'position_gain': [0 for _ in drivers],
        'avg_grid_last_5': [random.uniform(2, 8) for _ in drivers],
        'avg_position_last_5': [random.uniform(2, 9) for _ in drivers],
        'avg_points_last_5': [random.uniform(5, 25) for _ in drivers]
    })

    # Simulate 10 laps
    for lap in range(1, 11):
        print(f"\n Lap {lap} Simulation")
        f_out.write(f"\n Lap {lap} Simulation\n")

        # Randomly adjust position gains per lap
        driver_data['position_gain'] = driver_data['position_gain'] + \
            [random.choice([-1, 0, 1]) for _ in drivers]

        driver_data['position_gain'] = driver_data['position_gain'].clip(-5, 15)

        # Predict win probabilities
        features = ['grid', 'position_gain', 'avg_grid_last_5', 'avg_position_last_5', 'avg_points_last_5']
        X = driver_data[features]
        driver_data['win_probability'] = model.predict_proba(X)[:, 1]

        top_drivers = driver_data.sort_values(by='win_probability', ascending=False)
        result_str = top_drivers[['driver', 'grid', 'position_gain', 'win_probability']].to_string(index=False)

        print(result_str)
        f_out.write(result_str + "\n")

        time.sleep(2)

    f_out.write("\n Simulation complete!\n")
    print("\n Simulation complete! Results saved to 'results/lap_simulation_results.txt'")

