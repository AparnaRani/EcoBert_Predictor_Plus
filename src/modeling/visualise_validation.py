import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_model_name(name: str) -> str:
    """Cleans and standardizes model names."""
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

def visualize_best_validation_line():
    logger.info("Starting validation best results line chart visualization...")

    # === PATHS ===
    project_root = r"D:\EcoPredictor+"
    validation_path = os.path.join(project_root, "data", "validation", "comparison_results.csv")
    output_folder = os.path.join(project_root, "models")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(validation_path):
        logger.error("❌ comparison_results.csv not found. Please run run_prediction_on_validation.py first.")
        return

    # === LOAD VALIDATION RESULTS ===
    df = pd.read_csv(validation_path)
    logger.info(f"Loaded {len(df)} validation results.")

    # Normalize model names
    df["model_name"] = df["model_name"].apply(normalize_model_name)
    df["Abs_Error"] = df["Error_kg"].abs()

    # === Select Best Record (Lowest Abs Error) for Each Model ===
    best_per_model = df.loc[df.groupby("model_name")["Abs_Error"].idxmin()].reset_index(drop=True)
    best_per_model = best_per_model.sort_values(by="model_name")

    logger.info("✅ Best validation predictions per model:")
    logger.info(best_per_model[["model_name", "Actual_CO2", "Predicted_CO2", "Abs_Error"]])

    # === LINE PLOT ===
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    x_labels = best_per_model["model_name"].tolist()
    x = np.arange(len(x_labels))

    plt.plot(x, best_per_model["Actual_CO2"], marker="o", linewidth=2, label="Actual CO2", color="#5DADE2")
    plt.plot(x, best_per_model["Predicted_CO2"], marker="s", linewidth=2, label="Predicted CO2", color="#F5B041")

    plt.xticks(x, x_labels, rotation=25, ha="right")
    plt.ylabel("CO2 Emissions (kg)")
    plt.title("Validation Set: Predicted Sample per Model", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.5, linestyle="--")

    # Compact Y-axis for closeness
    y_min = 0
    y_max = best_per_model[["Actual_CO2", "Predicted_CO2"]].values.max() * 1.1
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(output_folder, "validation_best_per_model_line.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"✅ Validation line chart saved successfully at: {save_path}")

if __name__ == "__main__":
    visualize_best_validation_line()
