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

# -------------------------------------------------
# Model family extractor (supports new models)
# -------------------------------------------------
def extract_model_family(name: str) -> str:
    if not isinstance(name, str):
        return "other"

    base = name.split("/")[-1].lower()

    if "llama" in base:
        return "llama"
    if "qwen" in base:
        return "qwen"
    if "mistral" in base:
        return "mistral"
    if "mixtral" in base:
        return "mixtral"
    if "deepseek" in base:
        return "deepseek"
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


# -------------------------------------------------
# Main feature builder
# -------------------------------------------------
def build_features():
    logger.info("Starting feature engineering...")

    project_root = r"D:\EcoPredictor+"
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")

    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(os.path.join(raw_path, "cleaned_merged_data.csv"))
    logger.info(f"Loaded cleaned data: {df.shape}")

    if len(df) < 10:
        raise ValueError("Too few rows after cleaning. Aborting.")

    # ---------------- FEATURE ENGINEERING ----------------
    df["model_parameters"] = pd.to_numeric(df["model_parameters"], errors="coerce")
    df["log_model_parameters"] = np.log1p(df["model_parameters"])

    df["total_tokens"] = (
        df["num_train_samples"].clip(lower=1)
        * df["max_sequence_length"].clip(lower=1)
    )

    df["compute_load"] = (
        df["model_parameters"] * df["total_tokens"] * df["num_epochs"]
    )
    df["compute_log"] = np.log1p(df["compute_load"])

    df["size_cluster"] = np.where(df["model_parameters"] < 8e8, "small", "large")

    gpu_power_map = {
        "Tesla T4": 70,
        "TPU v2-8": 250,
        "Tesla P100-PCIE-16GB": 250,
        "NVIDIA GeForce RTX 3080": 320,
        "NVIDIA GeForce RTX 3090": 350,
        "NVIDIA V100": 300,
        "NVIDIA A100": 400,
        "NVIDIA H100": 700,
        "NVIDIA L4": 72,
    }

    df["gpu_power_watts"] = df["gpu_type"].map(gpu_power_map).fillna(200)
    df["model_family"] = df["model_name"].apply(extract_model_family)

    print("\nMODEL FAMILY DISTRIBUTION:")
    print(df["model_family"].value_counts())

    # ---------------- X / y ----------------
    X = df.drop(columns=["experiment_id", "CO2_emissions(kg)"])
    y = df["CO2_emissions(kg)"]

    # ---------------- TARGET TRANSFORM ----------------
    y_log = np.log1p(y)
    y_mean = y_log.mean()
    y_std = y_log.std()

    np.save(os.path.join(models_path, "target_mean.npy"), y_mean)
    np.save(os.path.join(models_path, "target_std.npy"), y_std)

    y_transformed = (y_log - y_mean) / y_std

    # ---------------- CORRECT SPLIT (ONLY 2 ARRAYS) ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_transformed,
        test_size=0.25,
        random_state=42
    )

    # Recover original-scale y_test using index alignment
    y_test_orig = y.loc[X_test.index].values

    # ---------------- PREPROCESSOR ----------------
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
    ]

    boolean = ["fp16"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_std", StandardScaler(), numeric_standard),
            ("num_q", QuantileTransformer(output_distribution="normal"), numeric_skewed),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("bool", OneHotEncoder(drop="if_binary"), boolean),
        ]
    )

    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_path, "preprocessor.joblib"))

    # ---------------- SAVE OUTPUTS ----------------
    X_train.to_csv(os.path.join(processed_path, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_path, "X_test_raw.csv"), index=False)

    pd.DataFrame({"y": y_train}).to_csv(
        os.path.join(processed_path, "y_train_transformed.csv"), index=False
    )
    pd.DataFrame({"y": y_test_orig}).to_csv(
        os.path.join(processed_path, "y_test_original.csv"), index=False
    )

    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    build_features()
