# run_single_heavy_exp.py
import os
import uuid
import pandas as pd
import torch
import eco2ai
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

def run_single_heavy_experiment(params: dict):
    """
    Runs one single, heavy experiment and logs the results to new, separate files.
    """
    experiment_id = str(uuid.uuid4())
    
    # --- Create the full metadata dictionary ---
    metadata = params.copy()
    metadata['experiment_id'] = experiment_id
    metadata['gpu_type'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    metadata['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Manually add the parameter count for the model being used
    metadata['model_parameters'] = 340000000 # For bert-large-uncased

    print(f"--- Starting Single Heavy Experiment ID: {experiment_id} ---")
    print(f"Parameters: {metadata}")

    # --- Define new, separate output files ---
    metadata_path = 'heavy_exp_1_metadata.csv'
    emissions_path = 'heavy_exp_1_emissions.csv'
    
    # Create a DataFrame with all columns, ensuring order
    all_columns = [
        'model_name','dataset_name','num_train_samples','num_epochs','batch_size',
        'fp16','pue','experiment_id','gpu_type','learning_rate','max_sequence_length',
        'gradient_accumulation_steps','num_gpus','dataset_config','model_parameters'
    ]
    metadata_df = pd.DataFrame([metadata], columns=all_columns)
    
    # Save the metadata, overwriting any previous test
    metadata_df.to_csv(metadata_path, index=False)

    tracker = eco2ai.Tracker(
        project_name="Heavy_LLM_Anchor_Run",
        experiment_description=f"run_{experiment_id}",
        file_name=emissions_path, # Log to the new file
    )
    tracker.start()

    try:
        model = AutoModelForSequenceClassification.from_pretrained(metadata['model_name'], num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(metadata['model_name'])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_dataset(metadata['dataset_name'], split=f"train[:{metadata['num_train_samples']}]")
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=metadata['max_sequence_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=f"./results/{experiment_id}",
            num_train_epochs=metadata['num_epochs'],
            per_device_train_batch_size=metadata['batch_size'],
            learning_rate=metadata['learning_rate'],
            gradient_accumulation_steps=metadata['gradient_accumulation_steps'],
            fp16=metadata['fp16'],
            save_strategy="no",
            report_to="none",
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
    except Exception as e:
        print(f"ERROR during experiment: {e}")
    finally:
        tracker.stop()
        print(f"--- Experiment Finished ---")

if __name__ == '__main__':
    # --- Define the parameters for this single, heavy run ---
    experiment_params = {
        'model_name': 'bert-large-uncased',
        'dataset_name': 'imdb',
        'num_train_samples': 25000,
        'num_epochs': 8,
        'batch_size': 16,
        'fp16': True,
        'pue': 1.58,
        'learning_rate': 2e-5,
        'max_sequence_length': 512,
        'gradient_accumulation_steps': 1,
        'dataset_config': None,
    }
    
    run_single_heavy_experiment(experiment_params)