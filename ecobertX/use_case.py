import json
import glob
import os

LOG_FOLDER = r"D:\EcoPredictor+\ecobertX\logs"

# ----------------------------------------------------
# 1. Load all prediction logs
# ----------------------------------------------------
files = glob.glob(os.path.join(LOG_FOLDER, "*.json"))

if not files:
    print("No logs found!")
    exit()

# ----------------------------------------------------
# 2. Select lowest-carbon configuration
# ----------------------------------------------------
best = None

for f in files:
    log = json.load(open(f))

    if not best or log["predicted_co2"] < best["predicted_co2"]:
        best = log
        best_file = f


print("\n🌱 ECOBERT-X LOG-BASED DECISION")
print("================================")
print("Selected log :", best_file)
print("Predicted CO₂:", round(best["predicted_co2"], 6), "kg")


# ----------------------------------------------------
# 3. Explain WHY this was low carbon
# ----------------------------------------------------
print("\nKey Reasons from Log:")

for r in best["shap_explanation"][:5]:

    direction = "reduces" if r["impact"] < 0 else "increases"

    print(f"- {r['feature']} {direction} CO₂ by {abs(r['impact']):.4f} kg")


# ----------------------------------------------------
# 4. Practical Usage Demonstration
# ----------------------------------------------------
print("\nHow this log is used in practice:")
print("""
• Before real training begins, several candidate configs are evaluated.
• EcoBERT-X predicts CO₂ for each and stores JSON logs.
• This script selects the lowest-emission option.
• SHAP reasons explain which hyperparameters caused the change.
• The team trains only the sustainable configuration.
""")
