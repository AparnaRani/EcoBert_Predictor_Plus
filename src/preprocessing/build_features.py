import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer # Added QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Added Pipeline for nested transformers
import joblib
import os
import numpy as np # Added for model_param_map and potential log transformation

def preprocess_data():
    print("Starting data preprocessing...")
    # Define paths
    raw_data_path = r'D:\EcoPredictor+\data\raw'
    processed_data_path = r'D:\EcoPredictor+\data\processed'
    models_path = r'D:\EcoPredictor+\models'

    # Ensure output directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # --- Define parameter counts for each model type ---
    # This dictionary should reflect the actual parameter counts for the models you used.
    # IMPORTANT: Ensure 'model_parameters' column is consistently named if it's already in your metadata.
    model_param_map = {
        'distilbert-base-uncased': 66_000_000,
        'bert-base-uncased': 110_000_000,
        'roberta-base': 125_000_000,
        't5-small': 60_000_000,
        'gpt2-xl': 1_500_000_000 # 1.5 Billion parameters
    }

    # Load data
    emissions_df = pd.read_csv(os.path.join(raw_data_path, 'emissions.csv'))
    metadata_df = pd.read_csv(os.path.join(raw_data_path, 'training_metadata.csv'))

    # Aggregate emissions data
    # NOTE: Assuming 'experiment_description' maps directly to an experiment ID or a unique run.
    # If your eco2ai log produces multiple entries per experiment, taking the max CO2 is a reasonable approach.
    emissions_agg = emissions_df.groupby('experiment_description')['CO2_emissions(kg)'].max().reset_index()
    emissions_agg['experiment_id'] = emissions_agg['experiment_description'].str.replace('run_', '') # Adjust if run_ ID format is different

    # Merge data
    df = pd.merge(metadata_df, emissions_agg[['experiment_id', 'CO2_emissions(kg)']], on='experiment_id')

    # --- NEW: Add 'model_parameters' feature ---
    # We will derive this from 'model_name' using our map.
    # Ensure 'model_name' column exists in your merged df.
    if 'model_name' in df.columns:
        df['model_parameters'] = df['model_name'].map(model_param_map)
        # Handle any model names not in your map (e.g., set to a default or drop rows)
        if df['model_parameters'].isnull().any():
            print("Warning: Some 'model_name' entries not found in 'model_param_map'. Consider handling them.")
            # Option: df = df.dropna(subset=['model_parameters']) # Drop rows with unknown models
            # Option: df['model_parameters'] = df['model_parameters'].fillna(some_default_value) # Fill with default
    else:
        print("Error: 'model_name' column not found in merged DataFrame. Cannot add 'model_parameters'.")
        return # Exit or raise error if 'model_name' is missing

    # --- NEW: Update features list to include 'model_parameters' ---
    target = 'CO2_emissions(kg)'
    features = [
        'model_parameters', # NEW KEY FEATURE
        'model_name',       # Keep as categorical for now, might be useful
        'num_train_samples',
        'num_epochs',
        'batch_size',
        'fp16',
        'pue',
        'gpu_type',
        # ADD ANY OTHER NUMERICAL FEATURES YOU MIGHT HAVE COLLECTED
        'learning_rate', # If you collected this
        'max_sequence_length', # If you collected this
        'gradient_accumulation_steps', # If you collected this
        'num_gpus' # Use 'num_gpus' from your metadata, not 'num_gpus_used'
    ]

    # Filter features to only those present in the DataFrame
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- NEW: Define preprocessing pipeline with specific treatment for 'model_parameters' ---
    # Separate numerical features that need log transform + scaling vs. just scaling
    numerical_features_log_scale = ['model_parameters']
    numerical_features_standard_scale = [f for f in X.select_dtypes(include=['int64', 'float64']).columns if f not in numerical_features_log_scale]

    # Handle boolean features (like 'fp16') by converting to int and including in standard scale
    boolean_features = X.select_dtypes(include=['bool']).columns
    for col in boolean_features:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        numerical_features_standard_scale.append(col)


    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('log_scale_params', Pipeline([
                # QuantileTransformer makes data Gaussian-like, good for models and handles outliers better than simple log.
                ('log', QuantileTransformer(output_distribution='normal', random_state=42)),
                ('scaler', StandardScaler())
            ]), numerical_features_log_scale),
            ('num', StandardScaler(), numerical_features_standard_scale),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Catches any features not explicitly handled, which is fine if no other features.
    )

    # Fit preprocessor and save it
    print("Fitting preprocessor...")
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_path, 'preprocessor.joblib'))
    print("Preprocessor fitted and saved.")

    # Save the raw splits for later transformation by train_model.py
    X_train.to_csv(os.path.join(processed_data_path, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)

    print("Data preprocessing complete. Raw splits saved.")

if __name__ == "__main__":
    preprocess_data()