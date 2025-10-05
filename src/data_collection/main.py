# src/data_collection/main.py
import itertools
# Import both GPU and TPU runners, even if only one is used
from .run_experiment import run_single_experiment as run_gpu_experiment
try:
    from .run_experiment_tpu import run_single_experiment as run_tpu_experiment
except ImportError:
    run_tpu_experiment = None # Handle case where TPU runner might not be available yet

# --- (All your previous get_..._batch functions can be here for reference if you have them) ---

# --- HIGH EMISSION ANCHOR BATCH ---
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
    RUN_GROUP = 'high_emission_anchor' # <-- SET THIS TO 'high_emission_anchor'
    
    print(f"Selected experiment group: {RUN_GROUP}")
    
    # Select the appropriate experiment runner
    experiment_runner = None
    if RUN_GROUP.startswith('tpu'):
        if run_tpu_experiment:
            experiment_runner = run_tpu_experiment
        else:
            print("ERROR: TPU RUNNER WAS SELECTED BUT src/data_collection/run_experiment_tpu.py IS NOT AVAILABLE OR HAS ERRORS.")
            return # Exit if TPU runner is not available but selected
    else:
        experiment_runner = run_gpu_experiment
    
    if experiment_runner is None:
        print("ERROR: No valid experiment runner could be determined. Please check RUN_GROUP and available scripts.")
        return

    group_map = {
        # ... (add any other experiment groups you have, like 'small_models_batch', 'medium_models_batch', etc.)
        'high_emission_anchor': get_high_emission_anchor_batch,
    }
    
    experiments_to_run = group_map.get(RUN_GROUP, lambda: [])()
        
    if not experiments_to_run:
        print(f"Warning: No experiments defined for group '{RUN_GROUP}'. Please check your group_map and batch functions.")
        return

    print(f"Total experiments to run: {len(experiments_to_run)}")
    
    for params in experiments_to_run:
        experiment_runner(params)
    
    print(f"All experiments for group '{RUN_GROUP}' have been completed.")

if __name__ == "__main__":
    main()