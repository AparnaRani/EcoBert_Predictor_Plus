# src/data_collection/main.py
from .run_experiment import run_single_experiment

# Group 1: For common GPUs like NVIDIA T4 (found on Colab)
experiments_for_T4 = [
    {'model_name': 'distilbert-base-uncased', 'dataset_name': 'imdb', 'num_train_samples': 10000, 'num_epochs': 1, 'batch_size': 32, 'fp16': True, 'pue': 1.58},
    {'model_name': 'bert-base-uncased', 'dataset_name': 'imdb', 'num_train_samples': 10000, 'num_epochs': 1, 'batch_size': 16, 'fp16': True, 'pue': 1.58},
]

# Group 2: For powerful GPUs like NVIDIA V100 (found on GCP)
experiments_for_V100 = [
    {'model_name': 'bert-base-uncased', 'dataset_name': 'imdb', 'num_train_samples': 50000, 'num_epochs': 2, 'batch_size': 32, 'fp16': True, 'pue': 1.3},
    {'model_name': 'roberta-base', 'dataset_name': 'imdb', 'num_train_samples': 50000, 'num_epochs': 2, 'batch_size': 32, 'fp16': True, 'pue': 1.3},
]

def main():
    print("Starting data collection run...")
    # --- CHOOSE WHICH GROUP TO RUN ---
    # Change this to `experiments_for_V100` when on a V100 machine.
    experiments_to_run = experiments_for_T4

    for params in experiments_to_run:
        run_single_experiment(params)

    print("All experiments in the group have been completed.")

if __name__ == "__main__":
    main()