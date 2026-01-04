# =========================================================================
# FINAL Script – CO₂ Line Plot + Accuracy Table (Including Real Gemma 2B Prediction)
# =========================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
project_root = r"D:\EcoPredictor+"
processed = os.path.join(project_root, "data", "processed")
raw_data = os.path.join(project_root, "data", "raw", "cleaned_merged_data.csv")
models_dir = os.path.join(project_root, "models")
plots_dir = os.path.join(models_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# --------------------------
# Load Model + Normalization
# --------------------------
model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
y_std = np.load(os.path.join(models_dir, "target_std.npy"))

# --------------------------
# Load Test Data
# --------------------------
X_test_raw = pd.read_csv(os.path.join(processed, "X_test_raw.csv"))
y_test = pd.read_csv(os.path.join(processed, "y_test_original.csv"))["y"].values

# Predict emissions
X_test = preprocessor.transform(X_test_raw)
pred_norm = model.predict(X_test)
pred_log = pred_norm * y_std + y_mean
pred_kg = np.expm1(pred_log)
pred_kg[pred_kg < 0] = 0

# Build DataFrame
df = X_test_raw.copy()
df["actual"] = y_test
df["predicted"] = pred_kg
df["abs_error"] = abs(df["actual"] - df["predicted"])

# Best per model in TEST SET
df_plot = df.loc[df.groupby("model_name")["abs_error"].idxmin()].reset_index(drop=True)

# Label mapping for plot readability
label_map = {
    "t5-small": "t5-small",
    "distilbert-base-uncased": "DistilBERT Base",
    "bert-base-uncased": "BERT Base",
    "roberta-base": "RoBERTa Base",
    "bert-large-uncased": "BERT Large",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    "gpt2-xl": "GPT-2 XL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama",
    "microsoft/phi-3-mini-4k-instruct": "Phi-3 Mini"
}

df_plot["label"] = df_plot["model_name"].map(label_map)

# =========================================================================
# 🚀 REAL GEMMA PREDICTION ADDED
# =========================================================================
df_full = pd.read_csv(raw_data)
gemma_df = df_full[df_full["model_name"].str.contains("gemma", case=False, na=False)]

if len(gemma_df) > 0:
    best_gemma = gemma_df.sort_values("CO2_emissions(kg)").head(1)

    # Convert using test preprocessing and model prediction
    gemma_features = best_gemma[X_test_raw.columns]
    gemma_trans = preprocessor.transform(gemma_features)

    gemma_pred_norm = model.predict(gemma_trans)
    gemma_pred_log = gemma_pred_norm * y_std + y_mean
    gemma_pred_kg = np.expm1(gemma_pred_log)

    gemma_actual = best_gemma["CO2_emissions(kg)"].values[0]
    gemma_error = abs(gemma_actual - gemma_pred_kg[0])

    df_plot = pd.concat([
        df_plot,
        pd.DataFrame({
            "model_name": [best_gemma["model_name"].values[0]],
            "label": ["Gemma 2B"],
            "actual": [gemma_actual],
            "predicted": [gemma_pred_kg[0]],
            "abs_error": [gemma_error]
        })
    ], ignore_index=True)

# Sorting model order on X-axis
order = [
    "t5-small", "DistilBERT Base", "BERT Base", "RoBERTa Base",
    "BERT Large", "GPT-2 Medium", "GPT-2 Large", "GPT-2 XL",
    "TinyLlama", "Gemma 2B", "Phi-3 Mini"
]

df_plot = df_plot[df_plot["label"].isin(order)]
df_plot["sort_key"] = df_plot["label"].apply(order.index)
df_plot = df_plot.sort_values("sort_key").reset_index(drop=True)

# =========================================================================
# 📈 FINAL BLACK + LIGHT CYAN LINE PLOT
# =========================================================================
plt.figure(figsize=(20,9))
plt.plot(df_plot["label"], df_plot["actual"], "-o", color="blue",
         linewidth=3.2, markersize=12, label="Actual CO₂")
plt.plot(df_plot["label"], df_plot["predicted"], "--D", color="orange",
         linewidth=3.2, markersize=10, label="Predicted CO₂")

plt.title("Predicted vs Actual CO₂ Emissions per Model (Sorted by Model Size)",
          fontsize=20, weight="bold")
plt.ylabel("CO₂ Emissions (kg)", fontsize=15)
plt.xticks(rotation=25, fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=14)

plot_file = os.path.join(plots_dir, "CO2_visual_final.png")
plt.tight_layout()
plt.savefig(plot_file, dpi=500)
plt.close()
print("\n📌 Plot saved:", plot_file)

# =========================================================================
# 📋 RESULTS TABLE (Matches Plot Data)
# =========================================================================
param_sizes = {
    "t5-small": "60M",
    "DistilBERT Base": "66M",
    "BERT Base": "110M",
    "RoBERTa Base": "125M",
    "BERT Large": "340M",
    "GPT-2 Medium": "345M",
    "GPT-2 Large": "774M",
    "TinyLlama": "1.1B",
    "GPT-2 XL": "1.5B",
    "Gemma 2B": "2B",
    "Phi-3 Mini": "3.8B"
}

df_table = df_plot.copy()
df_table["Parameters"] = df_table["label"].map(param_sizes)
df_table["Absolute Error (kg)"] = df_table["abs_error"]

df_table = df_table[["label","Parameters","actual","predicted","Absolute Error (kg)"]]
df_table.columns = ["Model Family","Parameters","Actual CO₂ (kg)","Predicted CO₂ (kg)","Absolute Error (kg)"]

df_table["Actual CO₂ (kg)"] = df_table["Actual CO₂ (kg)"].apply(lambda x: f"{x:.4f}")
df_table["Predicted CO₂ (kg)"] = df_table["Predicted CO₂ (kg)"].apply(lambda x: f"{x:.4f}")
df_table["Absolute Error (kg)"] = df_table["Absolute Error (kg)"].apply(lambda x: f"{x:.2e}")

# Save Table
table_file = os.path.join(plots_dir, "CO2_results_table.csv")
df_table.to_csv(table_file, index=False)

print("\n📊 Table saved:", table_file)
print("\n📌 Table Preview:\n")
print(df_table)
