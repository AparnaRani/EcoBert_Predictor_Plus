import pandas as pd
import numpy as np
import joblib
import os
import logging
import traceback # Import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_new_experiment():
    """
    Loads the saved model and predicts the emission for a
    new test experiment (bert-base-uncased).
    """
    logger.info("Starting prediction for new bert-base-uncased experiment...")

    # --- Define Paths ---
    project_root = r'D:\EcoPredictor+' 
    models_path = os.path.join(project_root, 'models')

    # --- Load Artifacts ---
    try:
        # 1. Load the trained model and preprocessor
        model = joblib.load(os.path.join(models_path, 'emission_predictor_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        logger.info("Loaded 'emission_predictor_model.joblib' and 'preprocessor.joblib'")

    except FileNotFoundError as e:
        logger.error(f"Error loading files: {e}.")
        logger.error("Please make sure you have run 'build_features.py' and 'train_model.py' successfully.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}")
        return

    # --- Define the new experiment's parameters ---
    # This is a quick, medium-sized run for 'bert-base-uncased'
    new_experiment_config = {
        'model_name': 'bert-base-uncased',
        'model_parameters': 109483778, # 110M
        'dataset_name': 'imdb',
        'pue': 1.58,
        'num_gpus': 1, # Assuming T4 or P100
        'gpu_type': 'P100', # Hardcode T4, as model was trained on it
        'dataset_config': np.nan,
        'num_epochs': 1,
        'fp16': True,
        'learning_rate': 2e-5,
        'max_sequence_length': 256,
        'num_train_samples': 15000,
        'batch_size': 16,
        'gradient_accumulation_steps': 2 # Effective batch size 32
    }
    
    # Convert the single experiment config into a DataFrame
    X_new = pd.DataFrame([new_experiment_config])
    
    try:
        # --- Run Prediction ---
        
        # 1. Apply preprocessor
        X_new_processed = preprocessor.transform(X_new)
        logger.info("Applied preprocessor to the new experiment data.")

        # 2. Predict on the log-scale
        prediction_log = model.predict(X_new_processed)
        
        # 3. Inverse-transform (np.log1p -> np.expm1) to get original kg scale
        prediction_kg = np.expm1(prediction_log[0])
        
        if prediction_kg < 0:
            prediction_kg = 0
            
        logger.info("Prediction complete.")

        # --- Display Results ---
        print("\n" + "="*40)
        print("   EcoBERT Predictor+ - New Model Forecast")
        print("="*40)
        print(f"  Model: {new_experiment_config['model_name']}")
        print(f"  Parameters: {new_experiment_config['model_parameters'] / 1_000_000:.1f} Million")
        print(f"  Training Samples: {new_experiment_config['num_train_samples']}")
        print(f"  Sequence Length: {new_experiment_config['max_sequence_length']}")
        print("="*40)
        print(f"  PREDICTED CO2 EMISSION: {prediction_kg:.6f} kg")
        print("="*40)
        print(f"(This is an estimate from your R²=0.8076 model.)\n")

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        logger.error("This may be due to a mismatch in expected columns or data.")
        traceback.print_exc()

if __name__ == "__main__":
    predict_new_experiment()

