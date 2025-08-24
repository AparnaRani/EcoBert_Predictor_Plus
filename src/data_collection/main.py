# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment

def get_colab_experiments():
    # ... (this function remains the same as before)
    models = ['distilbert-base-uncased', 'bert-base-uncased']
    datasets = ['imdb', 'sst2']
    num_samples = [5000, 15000, 25000]
    batch_sizes = [8, 16, 32]
    param_grid = itertools.product(models, datasets, num_samples, batch_sizes)
    experiments = []
    for model, dataset, samples, batch in param_grid:
        effective_samples = samples if dataset == 'imdb' else 67349 
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'num_train_samples': effective_samples,
            'num_epochs': 1, 'batch_size': batch, 'fp16': True, 'pue': 1.58
        })
    return experiments

def get_colab_varied_batch():
    """Group 2: A highly varied batch focusing on hyperparameters."""
    models = ['distilbert-base-uncased', 'albert-base-v2'] # Added a new efficient model
    num_samples = [20000] # Fixed sample size to focus on other params
    seq_lengths = [128, 256, 512]
    batch_sizes = [16, 32]
    learning_rates = [5e-5, 3e-5, 2e-5, 1e-5]
    # Total: 2 models * 1 sample * 3 seq_len * 2 batches * 4 lrs = 48 experiments
    
    param_grid = itertools.product(models, num_samples, seq_lengths, batch_sizes, learning_rates)
    
    experiments = []
    for model, samples, seq_len, batch, lr in param_grid:
        experiments.append({
            'model_name': model,
            'dataset_name': 'imdb', # Fixed dataset for this batch
            'num_train_samples': samples,
            'num_epochs': 1,
            'batch_size': batch,
            'max_sequence_length': seq_len, # New parameter
            'learning_rate': lr, # New parameter
            'fp16': True,
            'pue': 1.58
        })
    return experiments

# ... (get_kaggle_experiments and get_gcp_experiments functions remain the same)
def get_kaggle_experiments():
    # ...
    return []

def get_gcp_experiments():
    # ...
    return []


def main():
    """Selects and runs an experiment group."""
    # --- CHOOSE WHICH GROUP TO RUN ---
    # Change this value to 'colab', 'colab_varied', 'kaggle', or 'gcp'
    RUN_GROUP = 'colab_varied' # <-- SET TO RUN THE NEW BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    if RUN_GROUP == 'colab':
        experiments_to_run = get_colab_experiments()
    elif RUN_GROUP == 'colab_varied':
        experiments_to_run = get_colab_varied_batch()
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