import os
import joblib
import numpy as np
import pandas as pd

# Paths
project_root = r"D:\EcoPredictor+"
processed = os.path.join(project_root, "data", "processed")
models_dir = os.path.join(project_root, "models")

# Load data & model
X_test_raw = pd.read_csv(os.path.join(processed, "X_test_raw.csv"))
y_test = pd.read_csv(os.path.join(processed, "y_test_original.csv"))["y"]
model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
y_std = np.load(os.path.join(models_dir, "target_std.npy"))

# Model mapping used in final visual
target_models = [
    "t5-small",
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "bert-large-uncased",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-3-mini-4k-instruct"
]

results = []

for model_name in target_models:
    rows = X_test_raw[X_test_raw["model_name"] == model_name]
    
    if len(rows) == 0:
        print(f"⚠ {model_name} not found in test set!")
        continue
    
    idx = rows.index[0]  # Select first match = best used in plot
    row = rows.iloc[0:1]  # Single row dataframe
    
    actual = y_test.iloc[idx]

    # Predict using model
    X = preprocessor.transform(row)
    pred_norm = model.predict(X)
    pred_log = pred_norm * y_std + y_mean
    predicted = np.expm1(pred_log)[0]
    predicted = max(predicted, 0)

    row["actual_CO2(kg)"] = actual
    row["predicted_CO2(kg)"] = predicted

    results.append(row)

    print("\n==============================")
    print(f"MODEL: {model_name}")
    print("==============================")
    print(row.T)

# Save to CSV
final_df = pd.concat(results, ignore_index=True)
output_path = os.path.join(processed, "visual_models_full_rows.csv")
final_df.to_csv(output_path, index=False)

print("\n📌 Saved all 10 rows to:", output_path)
