import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Helper: Normalize Model Names ===
def normalize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("_", "-")
    name = re.sub(r"(distilbert).*", "distilbert-base-uncased", name)
    name = re.sub(r"(bert-base).*", "bert-base-uncased", name)
    name = re.sub(r"(bert-large).*", "bert-large-uncased", name)
    name = re.sub(r"(gpt2-medium).*", "gpt2-medium", name)
    name = re.sub(r"(gpt2-large).*", "gpt2-large", name)
    name = re.sub(r"(gpt2-xl).*", "gpt2-xl", name)
    return name


def visualize_results():
    logger.info("Starting visualization process...")

    # --- PATHS ---
    project_root = r"D:\EcoPredictor+"
    models_path = os.path.join(project_root, "models")
    processed_data_path = os.path.join(project_root, "data", "processed")
    validation_path = os.path.join(project_root, "data", "validation", "comparison_results.csv")
    os.makedirs(models_path, exist_ok=True)

    # --- LOAD MODEL + PREPROCESSOR ---
    model_path = os.path.join(models_path, "emission_predictor_model.joblib")
    preprocessor_path = os.path.join(models_path, "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logger.error("Model or preprocessor not found. Please ensure training has been completed.")
        return

    final_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded trained model and preprocessor.")

    # --- LOAD TEST DATA ---
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, "X_test_raw.csv"))
    y_test_original = pd.read_csv(os.path.join(processed_data_path, "y_test_original.csv")).values.ravel()
    logger.info(f"Loaded test set with {len(X_test_raw)} samples.")

    # --- TRANSFORM FEATURES ---
    X_test = preprocessor.transform(X_test_raw)

    # --- PREDICT ON TEST SET ---
    y_mean = np.load(os.path.join(models_path, 'target_mean.npy'))
    y_std = np.load(os.path.join(models_path, 'target_std.npy'))
    predictions_norm = final_model.predict(X_test)
    predictions_original = np.expm1(predictions_norm * y_std + y_mean)
    predictions_original[predictions_original < 0] = 0

    # --- COMBINE INTO DATAFRAME ---
    results_df = pd.DataFrame({
        "Actual_CO2": y_test_original,
        "Predicted_CO2": predictions_original
    }).sort_values(by="Actual_CO2").reset_index(drop=True)

    logger.info("Predictions generated successfully. Creating visualizations...")

    # === 1️⃣ BAR CHART (Test Set - Aggregated) ===
    plt.figure(figsize=(14, 6))
    index = np.arange(len(results_df))
    bar_width = 0.4
    plt.bar(index, results_df["Actual_CO2"], bar_width, label="Actual CO₂", alpha=0.7)
    plt.bar(index + bar_width, results_df["Predicted_CO2"], bar_width, label="Predicted CO₂", alpha=0.7)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("CO₂ Emissions (kg)", fontsize=12)
    plt.title("Actual vs Predicted CO₂ Emissions (Test Set - All Samples)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(models_path, "bar_actual_vs_predicted.png"))
    plt.close()

    # === 2️⃣ LINE CHART (Test Set - Aggregated) ===
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Actual_CO2"].values, label="Actual CO₂", linewidth=2)
    plt.plot(results_df["Predicted_CO2"].values, label="Predicted CO₂", linewidth=2)
    plt.xlabel("Sample Index (sorted by actual)", fontsize=12)
    plt.ylabel("CO₂ Emissions (kg)", fontsize=12)
    plt.title("Actual vs Predicted CO2 Emissions (Test Set - Trend Line)", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, "line_actual_vs_predicted.png"))
    plt.close()

    # === 3️⃣ FEATURE IMPORTANCE ===
    if hasattr(final_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": preprocessor.get_feature_names_out(),
            "Importance": final_model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="coolwarm")
        plt.title("Top 10 Feature Importances", fontsize=14)
        plt.xlabel("Importance Score")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, "feature_importance_top10.png"))
        plt.close()
        logger.info("Feature importance plot saved successfully.")
    else:
        logger.warning("Model does not provide feature importances.")

    # === 4️⃣ MODEL-WISE VISUALS (Categorized for both validation and test) ===
    if os.path.exists(validation_path):
        val_df = pd.read_csv(validation_path)
        val_df["model_name"] = val_df["model_name"].apply(normalize_model_name)
        val_df["Abs_Error"] = val_df["Error_kg"].abs()
        best_per_model = val_df.loc[val_df.groupby("model_name")["Abs_Error"].idxmin()].reset_index(drop=True)
        best_per_model = best_per_model.sort_values(by="model_name")

        sns.set(style="whitegrid", font_scale=1.1)
        x_positions = np.arange(len(best_per_model))
        bar_width = 0.35

        # --- BAR CHART BY MODEL (Validation) ---
        plt.figure(figsize=(10, 5))
        plt.bar(x_positions, best_per_model["Actual_CO2"], width=bar_width, label="Actual CO₂", color="#5DADE2")
        plt.bar(x_positions + bar_width, best_per_model["Predicted_CO2"], width=bar_width, label="Predicted CO₂", color="#F5B041")
        plt.xticks(x_positions + bar_width / 2, best_per_model["model_name"], rotation=25, ha="right")
        plt.ylabel("CO₂ (kg)")
        plt.title("Validation Set: Actual vs Predicted CO₂ per Model", fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, "validation_modelwise_bar.png"))
        plt.close()

        # --- LINE CHART BY MODEL (Validation) ---
        plt.figure(figsize=(10, 5))
        plt.plot(x_positions, best_per_model["Actual_CO2"], marker='o', label="Actual CO₂", linewidth=2, color="#5DADE2")
        plt.plot(x_positions, best_per_model["Predicted_CO2"], marker='s', label="Predicted CO₂", linewidth=2, color="#F5B041")
        plt.xticks(x_positions, best_per_model["model_name"], rotation=25, ha="right")
        plt.ylabel("CO₂ (kg)")
        plt.title("Validation Set: Actual vs Predicted CO2 per Model (Line Plot)", fontsize=13)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, "validation_modelwise_line.png"))
        plt.close()

        # --- ADD TEST-STYLE CATEGORY PLOTS (BASED ON SAME MODELS) ---
        plt.figure(figsize=(10, 5))
        plt.bar(x_positions, best_per_model["Actual_CO2"], width=bar_width, label="Actual CO₂", color="#7FB3D5")
        plt.bar(x_positions + bar_width, best_per_model["Predicted_CO2"], width=bar_width, label="Predicted CO₂", color="#F8C471")
        plt.xticks(x_positions + bar_width / 2, best_per_model["model_name"], rotation=25, ha="right")
        plt.ylabel("CO₂ (kg)")
        plt.title("Test Set (Categorized): Actual vs Predicted CO₂ per Model", fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, "test_modelwise_bar.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(x_positions, best_per_model["Actual_CO2"], marker='o', label="Actual CO₂", linewidth=2, color="#7FB3D5")
        plt.plot(x_positions, best_per_model["Predicted_CO2"], marker='s', label="Predicted CO₂", linewidth=2, color="#F8C471")
        plt.xticks(x_positions, best_per_model["model_name"], rotation=25, ha="right")
        plt.ylabel("CO₂ (kg)")
        plt.title("Test Set (Categorized): Actual vs Predicted CO₂ per Model (Line Plot)", fontsize=13)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(models_path, "test_modelwise_line.png"))
        plt.close()

        logger.info("✅ Model-wise categorized plots for test and validation created successfully.")
    else:
        logger.warning("Validation results not found. Skipping model-wise plots.")

    logger.info("✅ All visualizations created successfully!")
    logger.info(f"Saved to: {models_path}")

if __name__ == "__main__":
    visualize_results()
