# src/preprocessing/build_features.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def preprocess_data():
    print("Starting data preprocessing...")
    # Define paths
    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'
    models_path = 'models'

    # Ensure output directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Load data
    emissions_df = pd.read_csv(os.path.join(raw_data_path, 'emissions.csv'))
    metadata_df = pd.read_csv(os.path.join(raw_data_path, 'training_metadata.csv'))

    # Aggregate emissions data
    emissions_agg = emissions_df.groupby('experiment_description')['CO2_emissions(kg)'].max().reset_index()
    emissions_agg['experiment_id'] = emissions_agg['experiment_description'].str.replace('run_', '')

    # Merge data
    df = pd.merge(metadata_df, emissions_agg[['experiment_id', 'CO2_emissions(kg)']], on='experiment_id')

    target = 'CO2_emissions(kg)'
    features = ['model_name', 'num_train_samples', 'num_epochs', 'batch_size', 'fp16', 'pue', 'gpu_type']

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing pipeline
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit preprocessor and save it
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_path, 'preprocessor.joblib'))

    # Save the raw splits for later transformation
    X_train.to_csv(os.path.join(processed_data_path, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)

    print("Data preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()