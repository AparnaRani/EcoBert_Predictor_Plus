import os
import joblib
import numpy as np
import pandas as pd

# Path setup
project_root = r"D:\EcoPredictor+"
processed = os.path.join(project_root, "data", "processed")
models_dir = os.path.join(project_root, "models")

# Load files
X_test_raw = pd.read_csv(os.path.join(processed, "X_test_raw.csv"))
y_test = pd.read_csv(os.path.join(processed, "y_test_original.csv"))["y"].values
model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
y_std = np.load(os.path.join(models_dir, "target_std.npy"))

# Models we want
targets = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2-xl"
]

for target in targets:
    print("\n========================")
    print("MODEL:", target)
    print("========================")

    rows = X_test_raw[X_test_raw["model_name"] == target]
    if len(rows) == 0:
        print("❌ Model not found in test set")
        continue

    for idx, row in rows.iterrows():
        print("\n📌 Test Sample Index:", idx)

        # Extract original metadata
        print("\n--- Full X_test_raw row ---")
        print(row.to_frame().T)

        # Actual y value
        actual = y_test[idx]
        print("\nActual CO₂ (kg):", actual)

        # Prediction
        X = preprocessor.transform(row.to_frame())
        pred_norm = model.predict(X)
        pred_log = pred_norm * y_std + y_mean
        pred_kg = np.expm1(pred_log)[0]
        if pred_kg < 0:
            pred_kg = 0

        print("Predicted CO₂ (kg):", pred_kg)
        print("Absolute Error (kg):", abs(actual - pred_kg))
