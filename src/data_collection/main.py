# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment as run_gpu_experiment

# --- (All your previous get_..._batch functions are here) ---
def get_colab_experiments():
    return [] # Placeholder
def get_colab_varied_batch():
    return [] # Placeholder
def get_p100_batch_3():
    return [] # Placeholder
def get_gcp_experiments():
    return [] # Placeholder

# --- NEW MULTI-GPU BATCH ---
def get_multi_gpu_batch():
    """A batch designed for multi-GPU hardware like the T4 x2."""
    models = ['bert-base-uncased', 'roberta-base']
    num_samples = [50000] # A larger dataset to see the benefit
    batch_sizes = [32, 64] # Larger batch sizes are good for multi-GPU
    # Total: 2 models * 1 sample * 2 batches = 4 experiments
    
    param_grid = itertools.product(models, num_samples, batch_sizes)
    experiments = []
    for model, samples, batch in param_grid:
        experiments.append({
            'model_name': model, 'dataset_name': 'imdb', 'num_train_samples': samples,
            'num_epochs': 2, 'batch_size': batch, 'fp16': True, 'pue': 1.58
        })
    return experiments

def main():
    # --- CHOOSE WHICH GROUP TO RUN ---
    RUN_GROUP = 'multi_gpu_batch' # <-- SET TO RUN THE NEW BATCH
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    # We are only running GPU experiments now
    experiment_runner = run_gpu_experiment
    
    # NOTE: To run TPU batches again, you'll need the TPU code back in this file
    group_map = {
        'colab': get_colab_experiments,
        'colab_varied': get_colab_varied_batch,
        'p100_batch_3': get_p100_batch_3,
        'multi_gpu_batch': get_multi_gpu_batch, # <-- ADDED ENTRY
        'gcp': get_gcp_experiments,
    }
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()