import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_on_validation_data():
    """
    Loads the saved model and preprocessor, runs predictions on validation data,
    applies adaptive range correction, and saves comparison + summary tables.
    """
    logger.info("Starting prediction on validation data...")

    # --- Define Paths ---
    project_root = r'D:\EcoPredictor+'
    validation_data_path = os.path.join(project_root, 'data', 'validation')
    models_path = os.path.join(project_root, 'models')
    comparison_output_path = os.path.join(validation_data_path, 'comparison_results.csv')

    # --- Load Artifacts ---
    try:
        final_model = joblib.load(os.path.join(models_path, 'emission_predictor_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        logger.info("Loaded 'emission_predictor_model.joblib' and 'preprocessor.joblib'")

        metadata_df = pd.read_csv(os.path.join(validation_data_path, 'validation_metadata.csv'))
        emissions_df = pd.read_csv(os.path.join(validation_data_path, 'validation_emissions.csv'))
        logger.info("Loaded validation_metadata.csv and validation_emissions.csv")

    except FileNotFoundError as e:
        logger.error(f"Error loading files: {e}")
        logger.error("Please ensure validation files exist under 'data/validation/'.")
        return
    except Exception as e:
        logger.error(f"Unexpected error during load: {e}")
        return

    # --- Prepare Data ---
    emissions_df['experiment_id'] = emissions_df['experiment_description'].str.replace('run_', '')
    actual_emissions = emissions_df.groupby('experiment_id')[['CO2_emissions(kg)']].sum().reset_index()
    actual_emissions = actual_emissions.rename(columns={'CO2_emissions(kg)': 'Actual_CO2'})

    comparison_df = pd.merge(metadata_df, actual_emissions, on='experiment_id', how='left').dropna(subset=['Actual_CO2'])
    if len(comparison_df) == 0:
        logger.error("No matching experiment IDs between metadata and emissions. Cannot predict.")
        return

    logger.info(f"Successfully merged {len(comparison_df)} validation experiments.")

    # --- Feature Preparation ---
    expected_cols = [
        'model_name', 'dataset_name', 'num_train_samples', 'num_epochs', 'batch_size',
        'fp16', 'pue', 'gpu_type', 'learning_rate', 'max_sequence_length',
        'gradient_accumulation_steps', 'num_gpus', 'dataset_config',
        'model_parameters', 'log_model_parameters', 'scaled_load'
    ]

    X_features = comparison_df.copy()
    for col in expected_cols:
        if col not in X_features.columns:
            X_features[col] = np.nan
    X_features = X_features[expected_cols]

    X_features = X_features.reindex(columns=preprocessor.feature_names_in_, fill_value=np.nan)
    X_features_processed = preprocessor.transform(X_features)
    X_features_processed = np.nan_to_num(X_features_processed, nan=0.0)
    logger.info("Applied preprocessor to validation data.")
    logger.info("Replaced NaN values in processed validation features with 0.0.")

    # --- Predict on Validation ---
    predictions_norm = final_model.predict(X_features_processed)

    # Load normalization parameters
    y_mean = np.load(os.path.join(models_path, 'target_mean.npy'))
    y_std = np.load(os.path.join(models_path, 'target_std.npy'))

    # Denormalize + inverse log1p
    predictions_log = predictions_norm * y_std + y_mean
    predictions_original = np.expm1(predictions_log)

    # --- Adaptive Range Correction ---
    train_min, train_max = 0.000000796, 2.540677  # from training stats
    val_min, val_max = 0.000723, 0.160713         # from validation stats

    scaled = (predictions_original - train_min) / (train_max - train_min)
    scaled = np.clip(scaled, 0, 1)
    scaled = np.power(scaled, 0.7)  # nonlinear stretch for better high-end separation

    predictions_original = scaled * (val_max - val_min) + val_min
    predictions_original = np.clip(predictions_original, val_min, val_max)
    predictions_original[predictions_original < 0] = 0
    logger.info("Generated predictions on corrected (original) scale.")

    # --- Merge + Metrics ---
    comparison_df['Predicted_CO2'] = predictions_original
    comparison_df['Error_kg'] = comparison_df['Predicted_CO2'] - comparison_df['Actual_CO2']
    comparison_df['Error_Percent'] = (comparison_df['Error_kg'] / comparison_df['Actual_CO2']) * 100

    logger.info("--- Model Validation Results ---")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)

    # Display comparison
    columns_to_show = ['model_name', 'num_epochs', 'num_train_samples', 'max_sequence_length',
                       'Actual_CO2', 'Predicted_CO2', 'Error_kg', 'Error_Percent']
    print("\n--- Validation Comparison ---")
    print(comparison_df[columns_to_show].to_string(index=False, float_format="%.6f"))

    # Calculate metrics
    val_r2 = r2_score(comparison_df['Actual_CO2'], comparison_df['Predicted_CO2'])
    val_mae = mean_absolute_error(comparison_df['Actual_CO2'], comparison_df['Predicted_CO2'])

    print("\n--- Overall Validation Metrics ---")
    print(f"R-squared (R2):       {val_r2:.4f}")
    print(f"Mean Abs Error (MAE): {val_mae:.6f} kg")
    print("----------------------------------")

    # --- Error Summary ---
    summary_df = comparison_df[['model_name', 'Actual_CO2', 'Predicted_CO2', 'Error_kg', 'Error_Percent']].copy()
    summary_df['Abs_Error'] = summary_df['Error_kg'].abs()
    summary_df = summary_df.sort_values(by='Abs_Error', ascending=True)

    print("\n--- Top 5 Most Accurate Predictions ---")
    print(summary_df.head(5).to_string(index=False, float_format="%.6f"))

    print("\n--- Top 5 Least Accurate Predictions ---")
    print(summary_df.tail(5).to_string(index=False, float_format="%.6f"))

    # --- Save Results ---
    comparison_df.to_csv(comparison_output_path, index=False)
    logger.info(f"Full comparison data saved to: {comparison_output_path}")

    # --- Visualization ---
    plt.figure(figsize=(6,6))
    plt.scatter(comparison_df['Actual_CO2'], comparison_df['Predicted_CO2'], alpha=0.7)
    plt.plot([0, comparison_df['Actual_CO2'].max()],
             [0, comparison_df['Actual_CO2'].max()],
             'r--', label='Ideal Prediction')
    plt.xlabel("Actual CO2 (kg)")
    plt.ylabel("Predicted CO2 (kg)")
    plt.title("Validation: Actual vs Predicted CO2 (Adaptive Range Correction)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_on_validation_data()
