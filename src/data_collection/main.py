# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment as run_gpu_experiment
from .run_experiment_tpu import run_single_experiment as run_tpu_experiment

# --- GPU Experiment Groups ---
def get_colab_experiments():
    # ... (This function is already in your file)
    return []

def get_colab_varied_batch():
    # ... (This function is already in your file)
    return []

def get_gcp_experiments():
    # ... (This function is already in your file)
    return []

# --- TPU Experiment Groups ---
def get_tpu_batch_1():
    models = ['bert-base-uncased', 'roberta-base']
    num_samples = [30000]
    batch_sizes = [64, 128]
    seq_lengths = [128, 256]
    learning_rates = [5e-5, 3e-5]
    param_grid = itertools.product(models, num_samples, batch_sizes, seq_lengths, learning_rates)
    experiments = []
    for model, samples, batch, seq_len, lr in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 1, 'batch_size': batch, 'max_sequence_length': seq_len,
            'learning_rate': lr, 'bf16': True, 'pue': 1.1
        })
    return experiments

# --- THIS IS THE MISSING FUNCTION ---
def get_tpu_batch_2():
    """TPU Batch 2: Testing T5, a model well-suited for TPUs."""
    models = ['t5-small']
    num_samples = [50000, 100000]
    batch_sizes = [32, 64]
    seq_lengths = [256, 512]
    param_grid = itertools.product(models, num_samples, batch_sizes, seq_lengths)
    experiments = []
    for model, samples, batch, seq_len in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 1, 'batch_size': batch, 'max_sequence_length': seq_len, 
            'bf16': True, 'pue': 1.1
        })
    return experiments
# ------------------------------------

def main():
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'tpu_2' # <-- Make sure this is set to tpu_2
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    experiment_runner = run_tpu_experiment if RUN_GROUP.startswith('tpu') else run_gpu_experiment
    
    # --- ADD THE MISSING ENTRY TO THE MAP ---
    group_map = {
        'colab': get_colab_experiments,
        'colab_varied': get_colab_varied_batch,
        'tpu_1': get_tpu_batch_1,
        'tpu_2': get_tpu_batch_2, # <-- This was missing
        'gcp': get_gcp_experiments,
    }
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()