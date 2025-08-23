# src/modeling/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os

def train_and_evaluate():
    print("Starting model training and evaluation...")
    # Define paths
    processed_data_path = 'data/processed'
    models_path = 'models'

    # Load preprocessor and raw data splits
    preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
    X_train_raw = pd.read_csv(os.path.join(processed_data_path, 'X_train_raw.csv'))
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, 'X_test_raw.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()

    # Apply the loaded preprocessor
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kg CO2")
    print(f"Mean Absolute Error (MAE):    {mae:.4f} kg CO2")
    print(f"R-squared (R2):               {r2:.4f}")
    print("---------------------------------")

    # Save the trained model
    joblib.dump(model, os.path.join(models_path, 'emission_predictor_model.joblib'))
    print("Trained model saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()