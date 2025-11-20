import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_model_family(name: str) -> str:
    if not isinstance(name, str):
        return "other"
    base = name.split("/")[-1].lower()
    if "llama" in base:
        return "llama"
    if "gemma" in base:
        return "gemma"
    if "phi" in base:
        return "phi"
    if "bert" in base:
        return "bert"
    if base.startswith("gpt"):
        return "gpt"
    if base.startswith("t5"):
        return "t5"
    if "roberta" in base:
        return "roberta"
    return "other"

def build_features():
    logger.info("Starting feature engineering...")

    project_root = r"D:\EcoPredictor+"
    raw_data_path = os.path.join(project_root, "data", "raw")
    processed_data_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Load cleaned merged data
    df = pd.read_csv(os.path.join(raw_data_path, "cleaned_merged_data.csv"))
    logger.info(f"Loaded cleaned_merged_data.csv with shape {df.shape}")

    # --- Core numeric fields ---
    df["model_parameters"] = pd.to_numeric(df["model_parameters"], errors="coerce")
    df["log_model_parameters"] = np.log1p(df["model_parameters"])

    # Total tokens processed (very important)
    df["total_tokens"] = df["num_train_samples"] * df["max_sequence_length"]

    # Compute load ~ params * tokens * epochs
    df["compute_load"] = df["model_parameters"] * df["total_tokens"] * df["num_epochs"]
    df["compute_log"] = np.log1p(df["compute_load"])

    # Simple size cluster (can be used as categorical)
    df["size_cluster"] = np.where(df["model_parameters"] < 8e8, "small", "large")

    # GPU power (Watts) lookup based on your gpu_type list
    gpu_power_map = {
        "Tesla T4": 70,
        "TPU v2-8": 250,
        "Tesla P100-PCIE-16GB": 250,
        "NVIDIA GeForce RTX 3080": 320,
        "NVIDIA GeForce RTX 3090": 350
    }
    df["gpu_power_watts"] = df["gpu_type"].map(gpu_power_map).fillna(200)

    # Model family
    df["model_family"] = df["model_name"].apply(extract_model_family)

    # --- Define X and y ---
    if "experiment_id" not in df.columns or "CO2_emissions(kg)" not in df.columns:
        logger.error("Missing 'experiment_id' or 'CO2_emissions(kg)' in cleaned_merged_data.csv")
        return

    X = df.drop(columns=["experiment_id", "CO2_emissions(kg)"])
    y = df["CO2_emissions(kg)"]

    # --- Transform y (log + standardize) ---
    y_log = np.log1p(y)
    y_mean = y_log.mean()
    y_std = y_log.std()

    np.save(os.path.join(models_path, "target_mean.npy"), y_mean)
    np.save(os.path.join(models_path, "target_std.npy"), y_std)

    y_transformed = (y_log - y_mean) / y_std

    # Train / test split
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y_transformed, y, test_size=0.2, random_state=42
    )

    # --- Feature groups ---
    numeric_standard = [
        "num_train_samples",
        "num_epochs",
        "batch_size",
        "max_sequence_length",
        "learning_rate",
        "gradient_accumulation_steps",
        "num_gpus",
        "pue",
        "gpu_power_watts",
        "total_tokens",
    ]

    numeric_skewed = [
        "model_parameters",
        "log_model_parameters",
        "compute_log",
    ]

    categorical = [
        "model_family",
        "size_cluster",
        "dataset_name",
        "gpu_type",
        # you *can* add model_name, but it will explode dims; keeping it out helps generalization
    ]

    boolean = ["fp16"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_std", StandardScaler(), numeric_standard),
            ("num_quant", QuantileTransformer(output_distribution="normal"), numeric_skewed),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("bool", OneHotEncoder(drop="if_binary"), boolean),
        ],
        remainder="drop",
    )

    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_path, "preprocessor.joblib"))
    logger.info("Preprocessor fitted and saved.")

    # Save splits
    X_train.to_csv(os.path.join(processed_data_path, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_data_path, "X_test_raw.csv"), index=False)
    pd.DataFrame({"y": y_train}).to_csv(os.path.join(processed_data_path, "y_train_transformed.csv"), index=False)
    pd.DataFrame({"y": y_test}).to_csv(os.path.join(processed_data_path, "y_test_transformed.csv"), index=False)
    pd.DataFrame({"y": y_test_orig}).to_csv(os.path.join(processed_data_path, "y_test_original.csv"), index=False)

    logger.info("Feature engineering complete. Train/test splits saved.")

if __name__ == "__main__":
    build_features()
