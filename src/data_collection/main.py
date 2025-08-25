# src/data_collection/main.py
import itertools
# We now have two different experiment runners
from .run_experiment import run_single_experiment as run_gpu_experiment
from .run_experiment_tpu import run_single_experiment as run_tpu_experiment

# --- GPU Experiment Groups ---
def get_colab_experiments():
    models = ['distilbert-base-uncased', 'bert-base-uncased']
    datasets = ['imdb', 'sst2']
    num_samples = [5000, 15000, 25000]
    batch_sizes = [8, 16, 32]
    param_grid = itertools.product(models, datasets, num_samples, batch_sizes)
    experiments = []
    for model, dataset, samples, batch in param_grid:
        effective_samples = samples if dataset == 'imdb' else 67349 
        experiments.append({
            'model_name': model, 'dataset_name': dataset, 'num_train_samples': effective_samples,
            'num_epochs': 1, 'batch_size': batch, 'fp16': True, 'pue': 1.58
        })
    return experiments

def get_colab_varied_batch():
    models = ['distilbert-base-uncased', 'albert-base-v2']
    num_samples = [20000]
    seq_lengths = [128, 256, 512]
    batch_sizes = [16, 32]
    learning_rates = [5e-5, 3e-5, 2e-5, 1e-5]
    param_grid = itertools.product(models, num_samples, seq_lengths, batch_sizes, learning_rates)
    experiments = []
    for model, samples, seq_len, batch, lr in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 1, 'batch_size': batch, 'max_sequence_length': seq_len,
            'learning_rate': lr, 'fp16': True, 'pue': 1.58
        })
    return experiments

# --- TPU Experiment Groups ---
def get_tpu_batch_1():
    """TPU Batch 1 (Expanded): A wider variety of models and learning rates."""
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

def main():
    """Selects and runs an experiment group."""
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'tpu_1'  # <-- SET THIS TO RUN THE TPU BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    # Logic to select the right script (GPU vs TPU)
    if RUN_GROUP.startswith('tpu'):
        experiment_runner = run_tpu_experiment
    else:
        experiment_runner = run_gpu_experiment
    
    # Get the list of experiments
    if RUN_GROUP == 'colab':
        experiments_to_run = get_colab_experiments()
    elif RUN_GROUP == 'colab_varied':
        experiments_to_run = get_colab_varied_batch()
    elif RUN_GROUP == 'tpu_1':
        experiments_to_run = get_tpu_batch_1()
    else:
        experiments_to_run = [] # Add other groups here if needed
        
    print(f"Total experiments to run in this group: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()