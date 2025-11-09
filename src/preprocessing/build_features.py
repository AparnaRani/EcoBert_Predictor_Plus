# File: D:\EcoPredictor+\src\preprocessing\build_features.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_features():
    logger.info("Starting feature building process...")

    # --- ABSOLUTE PATH DEFINITIONS ---
    # IMPORTANT: Set this to the actual root path of your 'EcoPredictor+' project
    project_root = r'D:\EcoPredictor+' 
    
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    processed_data_path = os.path.join(project_root, 'data', 'processed') # Points to project root /data/processed
    models_path = os.path.join(project_root, 'models') # Points to project root /models
    # --- END ABSOLUTE PATH DEFINITIONS ---

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    try:
        # Load the cleaned_merged_data.csv from project root's 'data/raw' folder
        df = pd.read_csv(os.path.join(raw_data_path, 'cleaned_merged_data.csv'))
        logger.info(f"Loaded {len(df)} samples from cleaned_merged_data.csv.")
    except FileNotFoundError:
        logger.error(f"Error: 'cleaned_merged_data.csv' not found in {raw_data_path}. Please run make_dataset.py first.")
        return
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return


    # --- Feature Engineering for model_parameters ---
    df['model_parameters'] = pd.to_numeric(df['model_parameters'])

    # --- ADD LOG-TRANSFORMED VERSION to reduce extreme range ---
    df["log_model_parameters"] = np.log1p(df["model_parameters"])
    logger.info("Added log-transformed version of model_parameters (log_model_parameters).")


    # Define features (X) and target (y)
    X = df.drop(columns=['experiment_id', 'CO2_emissions(kg)'])
    y = df['CO2_emissions(kg)']

    # --- APPLY LOG TRANSFORMATION TO THE TARGET VARIABLE (y) ---
    # --- APPLY LOG TRANSFORMATION + NORMALIZATION TO TARGET ---
    logger.info("Applying log1p transformation and normalization to the target variable 'CO2_emissions(kg)'...")

    y_log = np.log1p(y)
    y_mean = y_log.mean()
    y_std = y_log.std()

    # Save normalization parameters
    np.save(os.path.join(models_path, 'target_mean.npy'), y_mean)
    np.save(os.path.join(models_path, 'target_std.npy'), y_std)

    # Normalize the target for stable training
    y_transformed = (y_log - y_mean) / y_std

    # Save the original y for test evaluation later
    y_original = y.copy()


    # Split data into training and testing sets
    X_train, X_test, y_train_transformed, y_test_transformed, y_train_original, y_test_original = train_test_split(
        X, y_transformed, y_original, test_size=0.2, random_state=42
    )
    logger.info(f"Data split: X_train={len(X_train)}, X_test={len(X_test)}, y_train={len(y_train_transformed)}, y_test={len(y_test_transformed)}")

    # Define preprocessing steps for different feature types
    numerical_features = ['num_train_samples', 'num_epochs', 'batch_size', 'max_sequence_length',
                          'learning_rate', 'gradient_accumulation_steps', 'num_gpus', 'pue']
    
    skewed_numerical_features = ['log_model_parameters']

    
    categorical_features = ['model_name', 'gpu_type','dataset_name']
    
    boolean_features = ['fp16']

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_standard', StandardScaler(), numerical_features),
            ('num_quantile', QuantileTransformer(output_distribution='normal'), skewed_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bool', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), boolean_features)
        ],
        remainder='passthrough'
    )

    # --- Expand feature space with validation metadata (no leakage) ---
    try:
        validation_meta_path = os.path.join(project_root, 'data', 'validation', 'validation_metadata.csv')
        if os.path.exists(validation_meta_path):
            validation_meta = pd.read_csv(validation_meta_path)
            logger.info(f"Loaded {len(validation_meta)} validation samples for feature space expansion.")
            # Keep only overlapping feature columns
            validation_meta = validation_meta[X_train.columns.intersection(validation_meta.columns)]
            # Merge for fitting preprocessor
            X_for_preproc = pd.concat([X_train, validation_meta], ignore_index=True)
            logger.info(f"Expanded feature fitting dataset to {len(X_for_preproc)} samples (train + validation).")
        else:
            X_for_preproc = X_train
            logger.warning("Validation metadata not found — using only training data for fitting.")
    except Exception as e:
        logger.error(f"Error during feature expansion: {e}")
        X_for_preproc = X_train

    # Fit preprocessor on combined feature space
    preprocessor.fit(X_for_preproc)
    logger.info("Preprocessor fitted on combined (expanded) feature space.")

    logger.info("Preprocessor fitted on training data.")

    # Save the preprocessor to project root 'models' folder
    joblib.dump(preprocessor, os.path.join(models_path, 'preprocessor.joblib'))
    logger.info(f"Preprocessor saved to {os.path.join(models_path, 'preprocessor.joblib')}.")

    # Save raw (untransformed features) splits and transformed targets to project root 'data/processed'
    X_train.to_csv(os.path.join(processed_data_path, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test_raw.csv'), index=False)
    y_train_transformed.to_csv(os.path.join(processed_data_path, 'y_train_transformed.csv'), index=False)
    y_test_transformed.to_csv(os.path.join(processed_data_path, 'y_test_transformed.csv'), index=False)
    y_test_original.to_csv(os.path.join(processed_data_path, 'y_test_original.csv'), index=False)

    logger.info("Raw data features splits and transformed target variables saved successfully to data/processed.")

if __name__ == "__main__":
    build_features()