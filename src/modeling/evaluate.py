import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_trained_model():
    logger.info("Starting evaluation of the trained model...")

    # === PATH SETUP ===
    project_root = r"D:\EcoPredictor+"
    models_path = os.path.join(project_root, "models")
    processed_data_path = os.path.join(project_root, "data", "processed")

    # === LOAD MODEL + PREPROCESSOR ===
    model_path = os.path.join(models_path, "emission_predictor_model.joblib")
    preprocessor_path = os.path.join(models_path, "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logger.error("❌ Model or preprocessor not found. Please ensure training was completed.")
        return

    final_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("✅ Loaded trained model and preprocessor.")

    # === LOAD TEST DATA ===
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, "X_test_raw.csv"))
    y_test_original = pd.read_csv(os.path.join(processed_data_path, "y_test_original.csv")).values.ravel()

    # === TRANSFORM FEATURES ===
    X_test = preprocessor.transform(X_test_raw)

    # === LOAD TARGET NORMALIZATION PARAMS ===
    y_mean = np.load(os.path.join(models_path, 'target_mean.npy'))
    y_std = np.load(os.path.join(models_path, 'target_std.npy'))

    # === PREDICT ===
    predictions_norm = final_model.predict(X_test)
    predictions_log = predictions_norm * y_std + y_mean
    predictions_original = np.expm1(predictions_log)
    predictions_original[predictions_original < 0] = 0

    # === EVALUATE ===
    r2 = r2_score(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))

    logger.info("\n--- 📊 Evaluation Results (on Test Set - Original Scale) ---")
    logger.info(f"R-squared (R²): {r2:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.6f} kg CO₂")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.6f} kg CO₂")
    logger.info("-----------------------------------------------------------")

if __name__ == "__main__":
    evaluate_trained_model()
