# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment

def get_colab_experiments():
    """Group 1: A broad set of runs for a standard T4 GPU."""
    models = ['distilbert-base-uncased', 'bert-base-uncased']
    datasets = ['imdb', 'sst2']
    num_samples = [5000, 15000, 25000]
    batch_sizes = [8, 16, 32]
    # Total: 2 models * 2 datasets * 3 samples * 3 batches = 36 experiments
    
    # Use itertools.product to create all combinations cleanly
    param_grid = itertools.product(models, datasets, num_samples, batch_sizes)
    
    experiments = []
    for model, dataset, samples, batch in param_grid:
        # SST2 is small, so we use all of it, not a subsample
        effective_samples = samples if dataset == 'imdb' else 67349 
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'num_train_samples': effective_samples,
            'num_epochs': 1,
            'batch_size': batch,
            'fp16': True,
            'pue': 1.58
        })
    return experiments

def get_kaggle_experiments():
    """Group 2: Focus on different optimizers and learning rates for T4/P100 GPUs."""
    models = ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base']
    datasets = ['imdb']
    num_samples = [10000, 20000]
    batch_sizes = [16, 32]
    # For this example, we'll keep other params fixed to vary optimizers/lr, but you can expand this
    # Total: 3 models * 1 dataset * 2 samples * 2 batches = 12 experiments. 
    # To get to ~50, you would add more parameter options here.
    
    param_grid = itertools.product(models, datasets, num_samples, batch_sizes)
    
    experiments = []
    for model, dataset, samples, batch in param_grid:
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'num_train_samples': samples,
            'num_epochs': 1,
            'batch_size': batch,
            'fp16': True,
            'pue': 1.58
        })
    # NOTE: You can expand this list with more loops for optimizers etc. to reach 50.
    return experiments

def get_gcp_experiments():
    """Group 3: Larger models and sample sizes for powerful V100/A100 GPUs."""
    models = ['bert-base-uncased', 'roberta-base']
    datasets = ['imdb']
    num_samples = [30000, 50000]
    batch_sizes = [16, 32, 64]
    num_epochs = [1, 2]
    # Total: 2 models * 1 dataset * 2 samples * 3 batches * 2 epochs = 24 experiments
    # NOTE: You can expand this list with more parameters to reach 50.
    
    param_grid = itertools.product(models, datasets, num_samples, batch_sizes, num_epochs)

    experiments = []
    for model, dataset, samples, batch, epochs in param_grid:
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'num_train_samples': samples,
            'num_epochs': epochs,
            'batch_size': batch,
            'fp16': True,
            'pue': 1.3 # Assuming a more efficient PUE for premium hardware
        })
    return experiments

def main():
    """
    Selects and runs an experiment group.
    """
    # --- CHOOSE WHICH GROUP TO RUN ---
    # Change this value to 'colab', 'kaggle', or 'gcp' depending on the platform.
    RUN_GROUP = 'colab'
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    if RUN_GROUP == 'colab':
        experiments_to_run = get_colab_experiments()
    elif RUN_GROUP == 'kaggle':
        experiments_to_run = get_kaggle_experiments()
    elif RUN_GROUP == 'gcp':
        experiments_to_run = get_gcp_experiments()
    else:
        raise ValueError("Invalid RUN_GROUP selected!")
        
    print(f"Total experiments to run in this group: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        run_single_experiment(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()