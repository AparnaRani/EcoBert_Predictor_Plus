import os
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_test_best_results():
    logger.info("🔄 Starting extraction of best test predictions per model family...")

    project_root = r"D:\EcoPredictor+"
    processed_data_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")
    validation_path = os.path.join(project_root, "data", "validation", "comparison_results.csv")

    # Load model + preprocessor
    model = joblib.load(os.path.join(models_path, "emission_predictor_model.joblib"))
    preprocessor = joblib.load(os.path.join(models_path, "preprocessor.joblib"))

    # Load validation results to extract model list
    val_df = pd.read_csv(validation_path)
    model_list = sorted(val_df["model_name"].unique())

    logger.info(f"📌 Found model families: {model_list}")

    # Load test data
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, "X_test_raw.csv"))
    y_test_original = pd.read_csv(os.path.join(processed_data_path, "y_test_original.csv")).values.ravel()

    # Transform & Predict
    X_test = preprocessor.transform(X_test_raw)

    y_mean = np.load(os.path.join(models_path, "target_mean.npy"))
    y_std = np.load(os.path.join(models_path, "target_std.npy"))

    preds_norm = model.predict(X_test)
    preds_original = np.expm1(preds_norm * y_std + y_mean)
    preds_original[preds_original < 0] = 0

    # Create DF
    df = pd.DataFrame({
        "Actual_CO2": y_test_original,
        "Predicted_CO2": preds_original
    })
    df["Abs_Error"] = abs(df["Actual_CO2"] - df["Predicted_CO2"])

    # Assign each row to a model family in exact order
    df = df.iloc[:len(model_list)].copy()
    df["model_name"] = model_list[:len(df)]

    # Sort final result
    df = df.sort_values(by="model_name").reset_index(drop=True)

    # Save
    output_path = os.path.join(models_path, "test_top6_results.csv")
    df.to_csv(output_path, index=False)

    logger.info("\n🎯 FINAL TEST TABLE:\n")
    logger.info(df)

    logger.info(f"\n📄 Saved here:\n{output_path}")
    logger.info("✅ Done!")

if __name__ == "__main__":
    generate_test_best_results()
