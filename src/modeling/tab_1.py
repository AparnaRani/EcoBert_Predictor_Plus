import os
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_best_test_table():
    logger.info("📌 Creating Best Test Prediction Table (one per model)...")

    project_root = r"D:\EcoPredictor+"
    models_path = os.path.join(project_root, "models")
    processed_path = os.path.join(project_root, "data", "processed")
    os.makedirs(models_path, exist_ok=True)

    # Load model + preprocessor
    model = joblib.load(os.path.join(models_path, "emission_predictor_model.joblib"))
    preprocessor = joblib.load(os.path.join(models_path, "preprocessor.joblib"))

    # Load test data
    df_test = pd.read_csv(os.path.join(processed_path, "X_test_raw.csv"))
    y_test_original = pd.read_csv(os.path.join(processed_path, "y_test_original.csv")).values.ravel()

    if "model_name" not in df_test.columns:
        logger.error("❌ 'model_name' not present in X_test_raw.csv")
        return

    # Predict + convert back to kg scale
    X_test = preprocessor.transform(df_test)
    y_mean = np.load(os.path.join(models_path, "target_mean.npy"))
    y_std = np.load(os.path.join(models_path, "target_std.npy"))
    pred_norm = model.predict(X_test)
    pred_orig = np.expm1(pred_norm * y_std + y_mean)
    pred_orig[pred_orig < 0] = 0  # safety

    # Build results table
    results = df_test.copy()
    results["Actual_CO2"] = y_test_original
    results["Predicted_CO2"] = pred_orig
    results["Abs_Error"] = abs(results["Actual_CO2"] - results["Predicted_CO2"])

    # Select the best prediction per model
    best_rows = results.loc[results.groupby("model_name")["Abs_Error"].idxmin()]
    best_rows = best_rows.sort_values("model_name")[["model_name", "Actual_CO2", "Predicted_CO2", "Abs_Error"]]

    # Save CSV
    table_path = os.path.join(models_path, "test_best_per_model_table.csv")
    best_rows.to_csv(table_path, index=False)

    logger.info("\n")
    logger.info(best_rows)
    logger.info(f"📄 Table saved: {table_path}")

if __name__ == "__main__":
    generate_best_test_table()
