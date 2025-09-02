# run_specific_scenario.py
from src.data_collection.run_experiment import run_single_experiment

# --- DEFINE THE EXACT SCENARIO YOU WANT TO MEASURE ---
# You can copy this from your predict.py script.
# NOTE: A large run like this might take a long time. 
# You can reduce 'num_train_samples' for a quicker test.
scenario_to_measure = {
    'model_name': 'roberta-base',
    'dataset_name': 'imdb',
    'dataset_config': None,
    'num_train_samples': 50000,
    'num_epochs': 2,
    'batch_size': 32,
    'fp16': True,
    'bf16': False, # Not used in the GPU script
    'pue': 1.58,
    'gpu_type': 'Tesla P100', # This will be auto-detected, but it's good practice to define
    'learning_rate': 3e-5,
    'max_sequence_length': 512,
    'gradient_accumulation_steps': 1,
    'num_gpus': 1 # This will be auto-detected
}
# ----------------------------------------------------

if __name__ == '__main__':
    print("--- Starting a single, measured experiment run ---")
    run_single_experiment(scenario_to_measure)
    print("\n--- Measurement run complete ---")
    print("Check your data/raw/emissions.csv and training_metadata.csv for the results.")