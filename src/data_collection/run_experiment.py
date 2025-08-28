# src/data_collection/run_experiment.py
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

def run_single_experiment(params: dict):
    experiment_id = str(uuid.uuid4())
    metadata = params.copy()
    metadata['experiment_id'] = experiment_id
    metadata['gpu_type'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"--- Starting Experiment ID: {experiment_id} ---")
    print(f"Parameters: {metadata}")

    metadata_df = pd.DataFrame([metadata])
    metadata_path = 'data/raw/training_metadata.csv'
    
    output_dir = os.path.dirname(metadata_path) 
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata (append if file exists)
    if not os.path.exists(metadata_path):
        metadata_df.to_csv(metadata_path, index=False)
    else:
        metadata_df.to_csv(metadata_path, mode='a', header=False, index=False)

    tracker = eco2ai.Tracker(
        project_name="EcoBERT_Predictor_Data_Collection",
        experiment_description=f"run_{experiment_id}",
        file_name="data/raw/emissions.csv",
        pue=metadata.get('pue', 1.58)
    )
    tracker.start()

    try:
        model = AutoModelForSequenceClassification.from_pretrained(metadata['model_name'], num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(metadata['model_name'])
        dataset = load_dataset(metadata['dataset_name'], split=f"train[:{metadata['num_train_samples']}]")
        
        # Use the new max_length from params
        seq_length = metadata.get('max_sequence_length', 512)
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=seq_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Use the new learning_rate from params
        lr = metadata.get('learning_rate', 2e-5)

        training_args = TrainingArguments(
            output_dir=f"./results/{experiment_id}",
            num_train_epochs=metadata['num_epochs'],
            per_device_train_batch_size=metadata['batch_size'],
            learning_rate=lr, # <-- ADDED
            gradient_accumulation_steps=metadata.get('gradient_accumulation_steps', 1), # <-- ADD THIS LINE
            fp16=metadata.get('fp16', False) and torch.cuda.is_available(),
            report_to="none",
            logging_steps=1000,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()
    except Exception as e:
        print(f"ERROR during experiment {experiment_id}: {e}")
    finally:
        tracker.stop()
        print(f"--- Finished Experiment ID: {experiment_id} ---")