import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re
import numpy as np
import time  # ✅ To prevent Windows save conflicts

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Normalize Model Names ---
def normalize_model_name(name: str) -> str:
    """
    Standardizes model names for grouping (e.g., bert-base-uncased, gpt2-xl, etc.)
    """
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

# --- Main Visualization Function ---
def visualize_test_predictions_by_model():
    logger.info("Starting line plot visualization for test predictions by model...")

    # === PATH SETUP ===
    project_root = r"D:\EcoPredictor+"
    validation_results_path = os.path.join(project_root, "data", "validation", "comparison_results.csv")
    output_folder = os.path.join(project_root, "models")
    os.makedirs(output_folder, exist_ok=True)

    # === LOAD RESULTS ===
    if not os.path.exists(validation_results_path):
        logger.error("❌ comparison_results.csv not found. Please run run_prediction_on_validation.py first.")
        return

    df = pd.read_csv(validation_results_path)
    logger.info(f"Loaded {len(df)} validation records.")

    # === CLEAN & PREPARE ===
    df["model_name"] = df["model_name"].apply(normalize_model_name)
    df["Abs_Error"] = df["Error_kg"].abs()

    # Select the best (lowest Abs_Error) prediction per model
    best_per_model = df.loc[df.groupby("model_name")["Abs_Error"].idxmin()].reset_index(drop=True)
    best_per_model = best_per_model.sort_values(by="model_name")

    logger.info("✅ Selected best prediction per model for line visualization.")
    logger.info(best_per_model[["model_name", "Actual_CO2", "Predicted_CO2", "Abs_Error"]])

    # === PLOT ===
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 5))

    # X positions for each model
    x = np.arange(len(best_per_model))

    # Plot Actual and Predicted as lines
    plt.plot(x, best_per_model["Actual_CO2"], marker='o', label="Actual CO₂", color="#5DADE2", linewidth=2)
    plt.plot(x, best_per_model["Predicted_CO2"], marker='s', label="Predicted CO₂", color="#F5B041", linewidth=2)

    # Beautify labels
    plt.xticks(x, best_per_model["model_name"], rotation=25, ha="right")
    plt.ylabel("CO₂ Emissions (kg)")
    plt.title("Test Predictions by Model Family - Actual vs Predicted CO₂", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Adjust Y scale (smooth, step = 0.01)
    y_min = 0
    y_max = max(best_per_model["Actual_CO2"].max(), best_per_model["Predicted_CO2"].max()) * 1.1
    plt.yticks(np.arange(y_min, y_max + 0.01, 0.01))

    plt.tight_layout()

    # === SAVE PLOT ===
    save_path = os.path.join(output_folder, "test_predictions_by_model_lineplot.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        time.sleep(0.5)  # tiny delay to ensure Windows file system sync
        logger.info(f"✅ Line plot successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"❌ Error saving plot: {e}")
    finally:
        plt.close()

    logger.info(f"Total models plotted: {len(best_per_model)}")
    logger.info("Visualization complete!")

# --- Run ---
if __name__ == "__main__":
    visualize_test_predictions_by_model()
