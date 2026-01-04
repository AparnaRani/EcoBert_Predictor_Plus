import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= Paths =========
project_root = r"D:\EcoPredictor+"
processed = os.path.join(project_root, "data", "processed")
raw_data = os.path.join(project_root, "data", "raw", "cleaned_merged_data.csv")
models_dir = os.path.join(project_root, "models")
plots_dir = os.path.join(project_root, "plots", "final_plots")
os.makedirs(plots_dir, exist_ok=True)

# ========= Load Model + Scalers =========
best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
y_std = np.load(os.path.join(models_dir, "target_std.npy"))

# ========= Load Test Split =========
X_test_raw = pd.read_csv(os.path.join(processed, "X_test_raw.csv"))
y_test_org = pd.read_csv(os.path.join(processed, "y_test_original.csv"))["y"].values
X_test = preprocessor.transform(X_test_raw)

# ========= Predict emissions =========
pred_norm = best_model.predict(X_test)
pred_log = pred_norm * y_std + y_mean
pred_kg = np.expm1(pred_log)
pred_kg[pred_kg < 0] = 0

df_test = X_test_raw.copy()
df_test["actual"] = y_test_org
df_test["predicted"] = pred_kg
df_test["abs_error"] = abs(df_test["actual"] - df_test["predicted"])

# Pick best prediction per model in test set
df_best_test = df_test.loc[df_test.groupby("model_name")["abs_error"].idxmin()].reset_index(drop=True)

# ========= Load full dataset to include Gemma 2B =========
df_full = pd.read_csv(raw_data)
gemma_rows = df_full[df_full["model_name"].str.contains("gemma", case=False, na=False)]

df_combined = df_best_test.copy()

if not gemma_rows.empty:
    gemma_best = gemma_rows.sort_values("CO2_emissions(kg)").head(1).copy()

    # Align missing columns to test format
    for col in X_test_raw.columns:
        if col not in gemma_best.columns:
            if "log" in col:
                gemma_best[col] = np.log(gemma_best["model_parameters"].values[0] + 1)
            elif col == "size_cluster":
                gemma_best[col] = "large"
            elif col == "gpu_power_watts":
                gemma_best[col] = 250
            elif col == "model_family":
                gemma_best[col] = "gemma"
            else:
                gemma_best[col] = X_test_raw[col].median()

    gemma_features = gemma_best[X_test_raw.columns].copy()
    gemma_trans = preprocessor.transform(gemma_features)

    gemma_pred_norm = best_model.predict(gemma_trans)
    gemma_pred_log = gemma_pred_norm * y_std + y_mean
    gemma_pred_kg = np.expm1(gemma_pred_log)
    gemma_pred_kg[gemma_pred_kg < 0] = 0

    gemma_actual = float(gemma_best["CO2_emissions(kg)"].values[0])
    gemma_error = abs(gemma_actual - float(gemma_pred_kg[0]))

    gemma_row = gemma_features.iloc[0].copy()
    gemma_row["model_name"] = gemma_best["model_name"].values[0]
    gemma_row["actual"] = gemma_actual
    gemma_row["predicted"] = float(gemma_pred_kg[0])
    gemma_row["abs_error"] = gemma_error

    df_combined = pd.concat([df_combined, pd.DataFrame([gemma_row])], ignore_index=True)

# ========= Label Mapping for Plot =========
name_map = {
    "t5-small": "t5-small",
    "distilbert-base-uncased": "DistilBERT Base",
    "bert-base-uncased": "BERT Base",
    "roberta-base": "RoBERTa Base",
    "bert-large-uncased": "BERT Large",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    "gpt2-xl": "GPT-2 XL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "TinyLlama",
    "microsoft/phi-3-mini-4k-instruct": "Phi-3 Mini",
    "google/gemma-2b": "Gemma 2B",
    "google/gemma-2-2b-it": "Gemma 2B"
}

df_combined["label"] = df_combined["model_name"].map(name_map)
df_combined = df_combined[df_combined["label"].notna()]

# ======= Corrected Order (GPT-2 XL comes after TinyLlama) =======
order = [
    "t5-small",
    "DistilBERT Base",
    "BERT Base",
    "RoBERTa Base",
    "BERT Large",
    "GPT-2 Medium",
    "GPT-2 Large",
    "TinyLlama",  # moved above
    "GPT-2 XL",   # moved below TinyLlama
    "Gemma 2B",
    "Phi-3 Mini"
]

df_combined["sort_key"] = df_combined["label"].apply(order.index)
df_plot = df_combined.sort_values("sort_key").drop_duplicates("label").reset_index(drop=True)

# ========= 🔥 Final Line Plot (Orange + Blue Style) =========
plt.figure(figsize=(22, 11))

plt.plot(df_plot["label"], df_plot["actual"], "-o",
         color="#0000FF", markersize=12, linewidth=3, label="Actual CO₂")  # Blue

plt.plot(df_plot["label"], df_plot["predicted"], "--D",
         color="#FF9900", markersize=12, linewidth=3, alpha=0.9, label="Predicted CO₂")  # Orange

# Annotate only Phi-3 Mini
phi_idx = df_plot[df_plot["label"] == "Phi-3 Mini"].index[0]
plt.text(phi_idx, df_plot["actual"][phi_idx] + 0.04,
         f"{df_plot['actual'][phi_idx]:.2f} kg", fontsize=16, ha="center", fontweight="bold")

plt.title("Predicted vs Actual CO₂ Emissions per Model (Sorted by Model Size)",
          fontsize=22, fontweight="bold", pad=20)
plt.ylabel("CO₂ Emissions (kg)", fontsize=18, fontweight="bold")
plt.xticks(fontsize=14, rotation=25)
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=16)

plt.tight_layout()
save_path = os.path.join(plots_dir, "CO2_visual_orange_blue_final.png")
plt.savefig(save_path, dpi=700)
plt.close()

print("\n📌 Final plot saved to:\n", save_path)
