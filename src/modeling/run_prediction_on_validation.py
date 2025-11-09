import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.metrics import r2_score, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_on_validation_data():
    """
    Loads the saved model and preprocessor, runs predictions on
    new validation data, and saves a comparison CSV.
    """
    logger.info("Starting prediction on validation data...")

    # --- Define Paths ---
    project_root = r'D:\EcoPredictor+' 
    validation_data_path = os.path.join(project_root, 'data', 'validation')
    models_path = os.path.join(project_root, 'models')
    
    # Define the final output file path
    comparison_output_path = os.path.join(validation_data_path, 'comparison_results.csv')

    # --- Load Artifacts ---
    try:
        # 1. Load the trained model and preprocessor
        final_model = joblib.load(os.path.join(models_path, 'emission_predictor_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        logger.info("Loaded 'emission_predictor_model.joblib' and 'preprocessor.joblib'")

        # 2. Load the NEW validation data
        metadata_df = pd.read_csv(os.path.join(validation_data_path, 'validation_metadata.csv'))
        emissions_df = pd.read_csv(os.path.join(validation_data_path, 'validation_emissions.csv'))
        logger.info("Loaded validation_metadata.csv and validation_emissions.csv")

    except FileNotFoundError as e:
        logger.error(f"Error loading files: {e}.")
        logger.error("Please run the Kaggle validation script and place output in 'D:\\EcoPredictor+\\data\\validation\\'.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return

    # --- Prepare Data for Comparison ---
    
    # 1. Extract experiment_id from emissions data
    emissions_df['experiment_id'] = emissions_df['experiment_description'].str.replace('run_', '')
    
    # 2. Merge actual emissions with metadata
    # We only care about the experiment_id and the actual CO2
    actual_emissions = emissions_df.groupby('experiment_id')[['CO2_emissions(kg)']].sum().reset_index()
    actual_emissions = actual_emissions.rename(columns={'CO2_emissions(kg)': 'Actual_CO2'})
    
    comparison_df = pd.merge(metadata_df, actual_emissions, on='experiment_id', how='left')
    
    # Handle any potential runs that metadata exists for but emissions failed to write (unlikely)
    comparison_df = comparison_df.dropna(subset=['Actual_CO2'])
    
    if len(comparison_df) == 0:
        logger.error("No matching experiment IDs found between metadata and emissions. Cannot predict.")
        return
        
    logger.info(f"Successfully merged {len(comparison_df)} validation experiments.")

    # 3. Define the features (X) for prediction
    # This must match the feature list from your build_features.py
    # We drop experiment_id and the target (Actual_CO2)
    X_features = comparison_df.drop(columns=['experiment_id', 'Actual_CO2'])
    
    # Ensure all expected columns are present, even if empty (like dataset_config)
    expected_cols = [
        'model_name', 'dataset_name', 'num_train_samples', 'num_epochs', 'batch_size',
        'fp16', 'pue', 'gpu_type', 'learning_rate', 'max_sequence_length',
        'gradient_accumulation_steps', 'num_gpus', 'dataset_config', 'model_parameters'
    ]
    for col in expected_cols:
        if col not in X_features.columns:
            X_features[col] = np.nan
            
    # Reorder columns to match the preprocessor's expectation
    X_features = X_features[expected_cols]

# --- Clip validation features to within training range ---
    try:
        train_data = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'cleaned_merged_data.csv'))
        for col in ['num_train_samples', 'num_epochs', 'batch_size', 'max_sequence_length', 'learning_rate']:
            if col in X_features.columns:
                min_val, max_val = train_data[col].min(), train_data[col].max()
                X_features[col] = X_features[col].clip(lower=min_val, upper=max_val)
        logger.info("Clipped validation feature ranges to match training limits.")
    except Exception as e:
        logger.warning(f"Could not clip feature ranges: {e}")

    # --- Run Prediction ---
    
    # --- Run Prediction ---

    # 0. Ensure correct column order and fill missing ones
    X_features = X_features.reindex(columns=preprocessor.feature_names_in_, fill_value=np.nan)
    logger.info("Reordered validation columns to match preprocessor expectations.")

    # 1. Apply preprocessor
    # The preprocessor (with SimpleImputer) will handle any NaNs
    X_features_processed = preprocessor.transform(X_features)
    logger.info("Applied preprocessor to validation data.")

    # --- Fix: Replace any NaNs (from unseen categories or missing data) ---
    
    X_features_processed = np.nan_to_num(X_features_processed, nan=0.0)
    logger.info("Replaced NaN values in processed validation features with 0.0.")


    # 2. Predict on the normalized log-scale
    predictions_norm = final_model.predict(X_features_processed)

    # 3. Load normalization parameters
    y_mean = np.load(os.path.join(models_path, 'target_mean.npy'))
    y_std = np.load(os.path.join(models_path, 'target_std.npy'))

    # 4. Denormalize (undo standardization)
    predictions_log = predictions_norm * y_std + y_mean

    # 5. Inverse log1p transform
    predictions_original = np.expm1(predictions_log)

    # 6. Fix negatives
    predictions_original[predictions_original < 0] = 0

    predictions_original[predictions_original < 0] = 0 # Handle negatives
    logger.info("Generated predictions on original scale.")

    # 4. Add predictions to our table
    comparison_df['Predicted_CO2'] = predictions_original
    
    # 5. Calculate the difference (error)
    comparison_df['Error_kg'] = comparison_df['Predicted_CO2'] - comparison_df['Actual_CO2']
    comparison_df['Error_Percent'] = (comparison_df['Error_kg'] / comparison_df['Actual_CO2']) * 100

    # --- Display & Save Results ---
    
    logger.info("--- Model Validation Results ---")
    
    # Set display options for a clean table
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)

    # Show the key columns
    columns_to_show = ['model_name', 'num_epochs', 'num_train_samples', 'max_sequence_length', 'Actual_CO2', 'Predicted_CO2', 'Error_kg', 'Error_Percent']
    print("\n--- Validation Comparison ---")
    print(comparison_df[columns_to_show].to_string(index=False, float_format="%.6f"))

    # Calculate overall metrics for this validation set
    val_r2 = r2_score(comparison_df['Actual_CO2'], comparison_df['Predicted_CO2'])
    val_mae = mean_absolute_error(comparison_df['Actual_CO2'], comparison_df['Predicted_CO2'])
    
    print("\n--- Overall Validation Metrics ---")
    print(f"R-squared (R2):       {val_r2:.4f}")
    print(f"Mean Abs Error (MAE): {val_mae:.6f} kg")
    print("----------------------------------")
    
    # Save the final comparison file
    try:
        comparison_df.to_csv(comparison_output_path, index=False)
        logger.info(f"Full comparison data saved to: {comparison_output_path}")
    except Exception as e:
        logger.error(f"Could not save comparison file: {e}")


if __name__ == "__main__":
    predict_on_validation_data()