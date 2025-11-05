import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score # Import R2 to display in title

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_visuals():
    """
    Loads the saved model and preprocessor to generate and display
    the final evaluation plots.
    """
    logger.info("Starting visualization generation...")

    # --- Define Paths ---
    project_root = r'D:\EcoPredictor+' 
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    models_path = os.path.join(project_root, 'models')

    # --- Load Artifacts ---
    try:
        # 1. Load the trained model
        final_model = joblib.load(os.path.join(models_path, 'emission_predictor_model.joblib'))
        model_name = type(final_model).__name__
        logger.info(f"Loaded '{model_name}' from 'emission_predictor_model.joblib'")

        # 2. Load the preprocessor
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        logger.info("Loaded 'preprocessor.joblib'")

        # 3. Load the raw test data and the original (non-transformed) test target
        X_test_raw = pd.read_csv(os.path.join(processed_data_path, 'X_test_raw.csv'))
        y_test_original = pd.read_csv(os.path.join(processed_data_path, 'y_test_original.csv')).values.ravel()
        logger.info("Loaded test data (X_test_raw.csv, y_test_original.csv)")

    except FileNotFoundError as e:
        logger.error(f"Error loading model artifacts: {e}. Please run train_model.py first.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return

    # --- Re-create Predictions ---
    
    # 1. Apply preprocessor
    X_test = preprocessor.transform(X_test_raw)
    logger.info("Applied preprocessor to X_test_raw.")

    # 2. Predict on the log-scale
    predictions_log = final_model.predict(X_test)
    
    # 3. Inverse-transform to get original kg scale
    predictions_original = np.expm1(predictions_log)
    predictions_original[predictions_original < 0] = 0 # Handle negatives
    logger.info("Generated predictions on original scale.")

    # 4. Calculate R2 for the plot title
    r2 = r2_score(y_test_original, predictions_original)
    logger.info(f"Loaded model {model_name} has R-squared: {r2:.4f}")

    # --- Generate Visualization (copied from train_model.py) ---
    logger.info("Generating prediction visualization...")
    plt.figure(figsize=(14, 6))

    # Plot 1: Actual vs. Predicted values (Original Scale)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test_original, y=predictions_original, alpha=0.6)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual CO2 Emissions (kg)", fontsize=12)
    plt.ylabel("Predicted CO2 Emissions (kg)", fontsize=12)
    # Add R2 to the title for clarity
    plt.title(f"Actual vs. Predicted (Test Set)\nModel: {model_name} (R2: {r2:.4f})", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Distribution of Residuals (Original Scale)
    plt.subplot(1, 2, 2)
    residuals = y_test_original - predictions_original
    sns.histplot(residuals, kde=True, bins=30, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.xlabel("Residuals (Actual - Predicted) (kg)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Residuals\nModel: {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout() 
    
    # Save a new copy of the plot
    plot_save_path = os.path.join(models_path, 'test_predictions_plot_regenerated.png')
    plt.savefig(plot_save_path) 
    logger.info(f"Prediction plot saved to {plot_save_path}.")
    
    # --- Display the plot ---
    logger.info("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    generate_visuals()
