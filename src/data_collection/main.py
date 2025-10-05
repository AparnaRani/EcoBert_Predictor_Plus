# src/data_collection/main.py
import itertools
from .run_experiment import run_single_experiment as run_gpu_experiment
try:
    from .run_experiment_tpu import run_single_experiment as run_tpu_experiment
except ImportError:
    run_tpu_experiment = None

def get_sequential_anchor_experiments():
    """Defines 10 individual, high-emission experiments for sequential data collection."""
    experiments = []
    
    # --- Batch 1: BERT-Large (6 experiments) ---
    bert_large_models = ['bert-large-uncased'] 
    bert_large_epochs = [3, 4, 5] 
    bert_large_batches = [16, 32]
    
    for model in bert_large_models:
        for epochs in bert_large_epochs:
            for batch in bert_large_batches:
                experiments.append({
                    'model_name': model,
                    'dataset_name': 'imdb',
                    'num_train_samples': 25000, # Full dataset
                    'num_epochs': epochs,
                    'batch_size': batch,
                    'fp16': True, 'pue': 1.58, 'learning_rate': 2e-5,
                    'max_sequence_length': 512, 'gradient_accumulation_steps': 1,
                    'dataset_config': None
                })

    # --- Batch 2: RoBERTa-Large (4 experiments) ---
    roberta_large_models = ['roberta-large']
    roberta_large_epochs = [3, 4] # Slightly fewer to keep total at 10
    roberta_large_batches = [16, 32]

    for model in roberta_large_models:
        for epochs in roberta_large_epochs:
            for batch in roberta_large_batches:
                experiments.append({
                    'model_name': model,
                    'dataset_name': 'imdb',
                    'num_train_samples': 25000,
                    'num_epochs': epochs,
                    'batch_size': batch,
                    'fp16': True, 'pue': 1.58, 'learning_rate': 2e-5,
                    'max_sequence_length': 512, 'gradient_accumulation_steps': 1,
                    'dataset_config': None
                })
                
    return experiments

def main():
    print("Preparing to run 10 high-emission anchor experiments sequentially.")
    
    experiments_to_run = get_sequential_anchor_experiments()
        
    if not experiments_to_run:
        print("Warning: No experiments defined for sequential anchor collection.")
        return

    print(f"Total experiments to run sequentially: {len(experiments_to_run)}")
    
    # We will use the GPU runner for these anchor points
    experiment_runner = run_gpu_experiment
    
    for i, params in enumerate(experiments_to_run):
        print(f"\n--- Running Experiment {i+1}/{len(experiments_to_run)} ---")
        experiment_runner(params)
        print(f"--- Experiment {i+1} Finished ---")
        
    print(f"\nAll {len(experiments_to_run)} high-emission anchor experiments have been completed.")

if __name__ == "__main__":
    main()