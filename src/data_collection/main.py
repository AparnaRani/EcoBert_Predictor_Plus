# src/data_collection/main.py
import itertools
# We only import the GPU runner by default. The TPU runner is imported conditionally.
from .run_experiment import run_single_experiment as run_gpu_experiment

# --- GPU Experiment Groups ---
def get_colab_experiments():
    """Batch 1: A basic set of experiments for initial data collection."""
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
    """Batch 2: A highly varied batch focusing on GPU hyperparameters."""
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

def get_gcp_experiments():
    """Batch for powerful GCP GPUs like the V100/A100."""
    models = ['bert-base-uncased', 'roberta-base']
    num_samples = [50000]
    batch_sizes = [32, 64]
    num_epochs = [1, 2]
    param_grid = itertools.product(models, num_samples, batch_sizes, num_epochs)
    experiments = []
    for model, samples, batch, epochs in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': epochs, 'batch_size': batch, 'fp16': True, 'pue': 1.3
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

def get_tpu_batch_2_expanded():
    """TPU Batch 2 (Expanded): Adds learning rates for more diversity."""
    models = ['t5-small']
    num_samples = [50000, 100000]
    batch_sizes = [32, 64]
    seq_lengths = [256, 512]
    learning_rates = [5e-5, 3e-5, 1e-5]
    param_grid = itertools.product(models, num_samples, batch_sizes, seq_lengths, learning_rates)
    experiments = []
    for model, samples, batch, seq_len, lr in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 1, 'batch_size': batch, 'max_sequence_length': seq_len, 
            'learning_rate': lr, 'bf16': True, 'pue': 1.1
        })
    return experiments
def get_p100_batch():
    """A focused batch for P100 GPUs testing gradient accumulation."""
    models = ['bert-base-uncased', 'roberta-base']
    num_samples = [30000]
    batch_sizes = [16, 32]
    gradient_accumulation_steps = [1, 2, 4]
    # Total: 2 models * 1 sample * 2 batches * 3 accum_steps = 12 experiments
    
    param_grid = itertools.product(models, num_samples, batch_sizes, gradient_accumulation_steps)
    experiments = []
    for model, samples, batch, accum_steps in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 1, 'batch_size': batch, 'max_sequence_length': 512,
            'gradient_accumulation_steps': accum_steps, 'fp16': True, 'pue': 1.58
        })
    return experiments

def main():
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'p100_batch'  # <-- SET TO RUN THE NEW BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    if RUN_GROUP.startswith('tpu'):
        from .run_experiment_tpu import run_single_experiment as run_tpu_experiment
        experiment_runner = run_tpu_experiment
    else:
        experiment_runner = run_gpu_experiment
    
    group_map = {
        'colab': get_colab_experiments,
        'colab_varied': get_colab_varied_batch,
        'tpu_1': get_tpu_batch_1,
        'tpu_2_expanded': get_tpu_batch_2_expanded,
        'p100_batch': get_p100_batch, # <-- ADDED ENTRY
        'gcp': get_gcp_experiments,
    }
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()