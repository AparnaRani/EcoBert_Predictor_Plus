'''# src/data_collection/main.py
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

def get_p100_batch_3():
    """Batch 3 for P100, focusing on a new dataset (GLUE/CoLA) and varying epochs."""
    models = ['bert-base-uncased', 'roberta-base']
    batch_sizes = [16, 32]
    num_epochs = [1, 2, 3]
    learning_rates = [5e-5, 3e-5]
    param_grid = itertools.product(models, batch_sizes, num_epochs, learning_rates)
    experiments = []
    for model, batch, epochs, lr in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'glue', 'dataset_config': 'cola',
            'num_train_samples': 8551, 'num_epochs': epochs, 'batch_size': batch,
            'learning_rate': lr, 'fp16': True, 'pue': 1.58
        })
    return experiments
    
def get_multi_gpu_batch():
    """A batch designed for multi-GPU hardware like the T4 x2."""
    models = ['bert-base-uncased', 'roberta-base']
    num_samples = [50000]
    batch_sizes = [32, 64]
    param_grid = itertools.product(models, num_samples, batch_sizes)
    experiments = []
    for model, samples, batch in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 2, 'batch_size': batch, 'fp16': True, 'pue': 1.58
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
def get_heavy_load_batch():
    """
    A computationally intensive batch designed to generate higher emission values
    by using a large model, the full dataset, and more epochs.
    """
    models = ['bert-large-uncased'] # A much larger model than bert-base
    num_epochs = [3, 4] # More training epochs
    batch_sizes = [16, 32] # Standard batch sizes for this model size
    
    # Total: 1 model * 2 epoch counts * 2 batch sizes = 4 long experiments
    param_grid = itertools.product(models, num_epochs, batch_sizes)
    experiments = []
    for model, epochs, batch in param_grid:
        experiments.append({
            'model_name': model,
            'dataset_name': 'imdb',
            'num_train_samples': 25000, # Use the full IMDb training set
            'num_epochs': epochs,
            'batch_size': batch,
            'fp16': True,
            'pue': 1.58, # PUE for Kaggle
            'max_sequence_length': 512 # Use the maximum sequence length
        })
    return experiments

def main():
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'heavy_load' # <-- SET TO RUN THE NEW HEAVY BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    # Conditionally import and select the correct runner (GPU vs TPU)
    if RUN_GROUP.startswith('tpu'):
        from .run_experiment_tpu import run_single_experiment as run_tpu_experiment
        experiment_runner = run_tpu_experiment
    else:
        experiment_runner = run_gpu_experiment
    
    # Map group names to functions
    group_map = {
        'colab': get_colab_experiments,
        'colab_varied': get_colab_varied_batch,
        'tpu_1': get_tpu_batch_1,
        'tpu_2_expanded': get_tpu_batch_2_expanded,
        'p100_batch_3': get_p100_batch_3,
        'multi_gpu_batch': get_multi_gpu_batch,
        'heavy_load': get_heavy_load_batch, # <-- ADDED ENTRY
    }
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()
'''

# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment as run_gpu_experiment
from .run_experiment_tpu import run_single_experiment as run_tpu_experiment

# --- (All your previous get_..._batch functions can be here for reference) ---

# --- NEW HIGH EMISSION ANCHOR BATCH ---
def get_high_emission_anchor_batch():
    """A batch designed to produce high emissions (>1kg) using a large model and many epochs."""
    models = ['bert-large-uncased'] # 340 Million parameters
    num_epochs = [3, 4, 5] # Train for a long time
    batch_sizes = [16, 32]
    # Total: 1 model * 3 epoch counts * 2 batches = 6 very long experiments
    
    param_grid = itertools.product(models, num_epochs, batch_sizes)
    experiments = []
    for model, epochs, batch in param_grid:
        experiments.append({
            'model_name': model,
            'dataset_name': 'imdb',
            'num_train_samples': 25000, # Use the full dataset
            'num_epochs': epochs,
            'batch_size': batch,
            'fp16': True,
            'pue': 1.58
        })
    return experiments

def main():
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'high_emission_anchor' # <-- SET TO RUN THE NEW BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    if RUN_GROUP.startswith('tpu'):
        from .run_experiment_tpu import run_single_experiment as run_tpu_experiment
        experiment_runner = run_tpu_experiment
    else:
        experiment_runner = run_gpu_experiment
    
    group_map = {
        # ... (all your previous entries can be here)
        'high_emission_anchor': get_high_emission_anchor_batch, # <-- ADDED ENTRY
    }
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()