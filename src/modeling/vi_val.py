import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re
import numpy as np  # ✅ Needed for tick range

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_model_name(name: str) -> str:
    """
    Cleans and standardizes model names (removes spaces, fixes capitalization).
    """
    name = str(name).strip().lower()
    # Remove duplicate spaces and unify formatting like "bert-base" or "gpt2-medium"
    name = re.sub(r"\s+", " ", name)
    name = name.replace("_", "-")
    name = re.sub(r"(distilbert).*", "distilbert-base-uncased", name)
    name = re.sub(r"(bert-base).*", "bert-base-uncased", name)
    name = re.sub(r"(bert-large).*", "bert-large-uncased", name)
    name = re.sub(r"(gpt2-medium).*", "gpt2-medium", name)
    name = re.sub(r"(gpt2-large).*", "gpt2-large", name)
    name = re.sub(r"(gpt2-xl).*", "gpt2-xl", name)
    return name

def visualize_best_per_model():
    logger.info("Starting visualization: best prediction per model family...")

    # === PATH SETUP ===
    project_root = r"D:\EcoPredictor+"
    validation_results_path = os.path.join(project_root, "data", "validation", "comparison_results.csv")
    output_folder = os.path.join(project_root, "models")
    os.makedirs(output_folder, exist_ok=True)

    # === LOAD VALIDATION RESULTS ===
    if not os.path.exists(validation_results_path):
        logger.error("❌ comparison_results.csv not found. Please run run_prediction_on_validation.py first.")
        return

    df = pd.read_csv(validation_results_path)
    logger.info(f"Loaded {len(df)} validation results.")

    # Normalize model names
    df["model_name"] = df["model_name"].apply(normalize_model_name)

    # Compute absolute error for accuracy sorting
    df["Abs_Error"] = df["Error_kg"].abs()

    # === Extract best (lowest error) record per unique model ===
    best_per_model = df.loc[df.groupby("model_name")["Abs_Error"].idxmin()].reset_index(drop=True)

    # Sort by model name for consistent order
    best_per_model = best_per_model.sort_values(by="model_name")

    logger.info("✅ Selected best prediction per model:\n")
    logger.info(best_per_model[["model_name", "Actual_CO2", "Predicted_CO2", "Abs_Error"]])

    # === VISUALIZATION ===
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    bar_width = 0.4
    x_positions = range(len(best_per_model))
    
    # Bar chart for actual vs predicted
    plt.bar(x_positions, best_per_model["Actual_CO2"], width=bar_width, label="Actual CO₂", color="#5DADE2")
    plt.bar([p + bar_width for p in x_positions], best_per_model["Predicted_CO2"], width=bar_width, label="Predicted CO₂", color="#F5B041")

    # Label and style
    plt.xticks([p + bar_width / 2 for p in x_positions], best_per_model["model_name"], rotation=30, ha="right")
    plt.ylabel("CO₂ Emissions (kg)")
    plt.title("Prediction per Model Family - Actual vs Predicted CO₂", fontsize=13)
    plt.legend()

    # === Set custom Y-axis tick spacing (increments of 0.01) ===
    y_min = 0
    y_max = max(best_per_model["Actual_CO2"].max(), best_per_model["Predicted_CO2"].max()) * 1.1
    plt.yticks(np.arange(y_min, y_max + 0.01, 0.01))  # ✅ increments of 0.01

    plt.tight_layout()

    # Save visualization
    save_path = os.path.join(output_folder, "best_per_model_validation_bar.png")
    plt.savefig(save_path)
    plt.close()

    logger.info(f"✅ Visualization saved to: {save_path}")
    logger.info(f"Total models plotted: {len(best_per_model)}")
    logger.info("Visualization complete!")

if __name__ == "__main__":
    visualize_best_per_model()
