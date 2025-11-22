import joblib
import numpy as np
import pandas as pd
import math

# Load artifacts
preprocessor = joblib.load(r"D:\EcoPredictor+\models\preprocessor.joblib")
model = joblib.load(r"D:\EcoPredictor+\models\best_model.joblib")
y_mean = np.load(r"D:\EcoPredictor+\models\target_mean.npy")
y_std = np.load(r"D:\EcoPredictor+\models\target_std.npy")

# GPU power lookup (extend later)
gpu_power_map = {
    "Tesla T4": 70,
    "Tesla P100-PCIE-16GB": 250,
    "Tesla V100": 300,
    "NVIDIA GeForce RTX 3080": 320,
    "NVIDIA GeForce RTX 3090": 350,
}

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived fields consistent with training
    df["log_model_parameters"] = np.log1p(df["model_parameters"])
    
    df["total_tokens"] = (
        df["num_train_samples"] 
        * df["max_sequence_length"]
        * df["num_epochs"]
    )
    
    df["compute_log"] = np.log1p(
        df["total_tokens"] * df["model_parameters"]
    )

    df["gpu_power_watts"] = df["gpu_type"].map(gpu_power_map).fillna(250)

    # Cluster size: GPUs × grad_accum
    df["size_cluster"] = df["num_gpus"] * df["gradient_accumulation_steps"]

    # Model Family extraction
    df["model_family"] = df["model_name"].apply(lambda x: x.split('/')[0] if '/' in x else x)

    return df


def predict_emissions(run_config: dict) -> float:
    df = pd.DataFrame([run_config])

    df = add_engineered_features(df)   # ⬅ FIXED: Add missing features here

    X = preprocessor.transform(df)

    pred_norm = model.predict(X)
    pred_log = pred_norm * y_std + y_mean
    pred_kg = np.expm1(pred_log)[0]

    return max(pred_kg, 0)


# Test config
example = {
    "model_name": "microsoft/phi-3-mini-4k-instruct",
    "dataset_name": "imdb",
    "num_train_samples": 20000,
    "num_epochs": 1,
    "batch_size": 1,
    "fp16": False,
    "pue": 1.58,
    "learning_rate": 2e-5,
    "max_sequence_length": 1024,
    "gradient_accumulation_steps": 32,
    "num_gpus": 1,
    "gpu_type": "Tesla P100-PCIE-16GB",
    "model_parameters": 3800000000,
}

pred = predict_emissions(example)
print(f"🔥 Predicted CO₂ emissions: {pred:.3f} kg")
