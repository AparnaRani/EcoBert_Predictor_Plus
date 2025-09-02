# src/modeling/predict.py
import pandas as pd
import joblib
import warnings
import os

# Suppress warnings from scikit-learn about feature names
warnings.filterwarnings("ignore", category=UserWarning)

def predict_emissions(input_data: dict):
    """
    Loads the trained model and preprocessor to predict CO2 emissions for a new set of parameters.
    """
    models_path = 'models'
    try:
        model = joblib.load(os.path.join(models_path, 'emission_predictor_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        print("✅ Model and preprocessor loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Model or preprocessor not found. Please run the training pipeline first.")
        return None

    input_df = pd.DataFrame([input_data])
    
    print("⚙️  Preprocessing input features...")
    input_processed = preprocessor.transform(input_df)

    print("🔮 Making prediction...")
    prediction = model.predict(input_processed)
    
    predicted_co2 = prediction[0]
    
    return predicted_co2

if __name__ == '__main__':
    # --- DEFINE YOUR SCENARIO TO PREDICT HERE ---
    scenario_1 = {
        'model_name': 'roberta-base',
        'dataset_name': 'imdb',
        'dataset_config': None,
        'num_train_samples': 50000,
        'num_epochs': 2,
        'batch_size': 32,
        'fp16': True,
        'bf16': False,
        'pue': 1.4,
        # Corrected to a GPU the model was actually trained on
        'gpu_type': 'Tesla P100', 
        'learning_rate': 3e-5,
        'max_sequence_length': 512,
        'gradient_accumulation_steps': 1,
        'num_gpus': 1
    }
    # ---------------------------------------------
    
    print("\n--- EcoBERT Predictor+ ---")
    print(f"Calculating emissions for the defined scenario...")
    
    predicted_emissions = predict_emissions(scenario_1)

    if predicted_emissions is not None:
        print("\n--- PREDICTION RESULT ---")
        print(f"Predicted CO2 Emissions: {predicted_emissions:.4f} kg")
        print("-------------------------\n")