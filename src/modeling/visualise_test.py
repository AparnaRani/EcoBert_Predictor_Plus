import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def visualize_best_test_predictions():
    logger.info("Starting best test prediction visualization (one per model)...")

    # --- PATHS ---
    project_root = r"D:\EcoPredictor+"
    models_path = os.path.join(project_root, "models")
    processed_path = os.path.join(project_root, "data", "processed")

    os.makedirs(models_path, exist_ok=True)

    # --- LOAD MODEL + PREPROCESSOR ---
    model_path = os.path.join(models_path, "emission_predictor_model.joblib")
    preprocessor_path = os.path.join(models_path, "preprocessor.joblib")

    if not os.path.exists(model_path):
        logger.error("❌ Model not found. Please run train_and_evaluate.py first.")
        return

    final_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # --- LOAD TEST DATA ---
    X_test_raw = pd.read_csv(os.path.join(processed_path, "X_test_raw.csv"))
    y_test_original = pd.read_csv(os.path.join(processed_path, "y_test_original.csv")).values.ravel()

    # Ensure model_name column exists in test metadata
    if "model_name" not in X_test_raw.columns:
        logger.error("❌ 'model_name' column not found in X_test_raw.csv.")
        return

    # --- TRANSFORM AND PREDICT ---
    X_test = preprocessor.transform(X_test_raw)

    y_mean = np.load(os.path.join(models_path, "target_mean.npy"))
    y_std = np.load(os.path.join(models_path, "target_std.npy"))

    predictions_norm = final_model.predict(X_test)
    predictions_original = np.expm1(predictions_norm * y_std + y_mean)
    predictions_original[predictions_original < 0] = 0

    # --- CREATE RESULTS DATAFRAME ---
    results = X_test_raw.copy()
    results["Actual_CO2"] = y_test_original
    results["Predicted_CO2"] = predictions_original
    results["Error_kg"] = results["Predicted_CO2"] - results["Actual_CO2"]
    results["Abs_Error"] = np.abs(results["Error_kg"])

    # --- SELECT BEST (LOWEST ERROR) SAMPLE PER MODEL ---
    best_per_model = results.loc[results.groupby("model_name")["Abs_Error"].idxmin()].reset_index(drop=True)
    best_per_model = best_per_model.sort_values(by="model_name")

    logger.info("✅Test predictions per model selected:")
    logger.info(best_per_model[["model_name", "Actual_CO2", "Predicted_CO2", "Abs_Error"]])

    # --- VISUALIZATION (LINE CHART) ---
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    x_labels = best_per_model["model_name"].tolist()
    x = np.arange(len(x_labels))

    plt.plot(x, best_per_model["Actual_CO2"], marker="o", linewidth=2, label="Actual CO2", color="#5DADE2")
    plt.plot(x, best_per_model["Predicted_CO2"], marker="s", linewidth=2, label="Predicted CO2", color="#F5B041")

    plt.xticks(x, x_labels, rotation=25, ha="right")
    plt.ylabel("CO2 Emissions (kg)")
    plt.title("Test Set: Predicted Sample per Model", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.5, linestyle="--")

    # Compact y-axis for visual closeness
    y_min = 0
    y_max = best_per_model[["Actual_CO2", "Predicted_CO2"]].values.max() * 1.1
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(models_path, "test_best_per_model_line.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"✅ Line chart saved successfully at: {save_path}")

if __name__ == "__main__":
    visualize_best_test_predictions()
