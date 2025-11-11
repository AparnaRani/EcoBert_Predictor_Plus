import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_all_models():
    logger.info("📊 Starting visualization of all model metrics...")

    # === PATH SETUP ===
    project_root = r"D:\EcoPredictor+"
    models_path = os.path.join(project_root, "models")
    results_csv = os.path.join(models_path, "all_model_results.csv")

    # === CHECK IF RESULTS FILE EXISTS ===
    if not os.path.exists(results_csv):
        logger.error("❌ 'all_model_results.csv' not found. Please run train_and_evaluate.py first.")
        return

    # === LOAD RESULTS ===
    df = pd.read_csv(results_csv)
    logger.info(f"✅ Loaded results for {len(df)} models.")
    logger.info(f"\n{df}")

    # === STYLE SETUP ===
    sns.set(style="whitegrid", font_scale=1.1)
    color_palette = sns.color_palette("viridis", len(df))

    # === 1️⃣ RMSE COMPARISON ===
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="RMSE", data=df, palette=color_palette)
    best_rmse_model = df.loc[df['RMSE'].idxmin(), 'Model']
    plt.title(f"Model Comparison - RMSE (Lower = Better)\n🏆 Best: {best_rmse_model}", fontsize=14)
    plt.ylabel("Root Mean Squared Error (kg CO₂)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, "model_rmse_comparison.png"))
    plt.close()

    # === 2️⃣ MAE COMPARISON ===
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="MAE", data=df, palette="crest")
    best_mae_model = df.loc[df['MAE'].idxmin(), 'Model']
    plt.title(f"Model Comparison - MAE (Lower = Better)\n🏆 Best: {best_mae_model}", fontsize=14)
    plt.ylabel("Mean Absolute Error (kg CO₂)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, "model_mae_comparison.png"))
    plt.close()

    # === 3️⃣ R² COMPARISON ===
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="R2", data=df, palette="coolwarm")
    best_r2_model = df.loc[df['R2'].idxmax(), 'Model']
    plt.title(f"Model Comparison - R² (Higher = Better)\n🏆 Best: {best_r2_model}", fontsize=14)
    plt.ylabel("R² Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, "model_r2_comparison.png"))
    plt.close()

    # === 4️⃣ COMBINED OVERVIEW (RMSE, MAE, R² TOGETHER) ===
    df_melted = df.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"], var_name="Metric", value_name="Value")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_melted, palette="mako")
    plt.title("Overall Model Performance Overview", fontsize=14)
    plt.xticks(rotation=30)
    plt.legend(title="Metric", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, "model_performance_overview.png"))
    plt.close()

    logger.info("✅ Visualization complete! All plots saved successfully.")
    logger.info(f"📁 Saved to: {models_path}")

if __name__ == "__main__":
    visualize_all_models()
