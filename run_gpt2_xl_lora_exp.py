# run_gpt2_xl_lora_exp.py
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
from peft import LoraConfig, get_peft_model, TaskType

def run_gpt2_xl_lora_experiment(params: dict):
    """
    Runs a single experiment fine-tuning GPT2-XL with LoRA, logging results to separate files.
    """
    experiment_id = str(uuid.uuid4())
    
    metadata = params.copy()
    metadata['experiment_id'] = experiment_id
    metadata['gpu_type'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    metadata['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # --- CRITICAL: Set model_parameters for gpt2-xl ---
    metadata['model_parameters'] = 1500000000 # 1.5 Billion parameters for gpt2-xl

    print(f"--- Starting GPT2-XL LoRA Experiment ID: {experiment_id} ---")
    print(f"Parameters: {metadata}")

    # --- Define NEW, SEPARATE output files for this experiment ---
    # These files will NOT be appended to your existing EcoBERT Predictor+ dataset.
    # You will manually merge them later.
    metadata_path = 'gpt2_xl_lora_metadata.csv'
    emissions_path = 'gpt2_xl_lora_emissions.csv'
    
    # Define all expected columns to ensure consistency in the output CSV
    all_columns = [
        'model_name', 'dataset_name', 'num_train_samples', 'num_epochs', 'batch_size',
        'fp16', 'pue', 'experiment_id', 'gpu_type', 'learning_rate', 'max_sequence_length',
        'gradient_accumulation_steps', 'num_gpus', 'dataset_config', 'model_parameters'
    ]
    
    # Write metadata to the new CSV (overwrites if exists, which is fine for a single run)
    metadata_df = pd.DataFrame([metadata], columns=all_columns)
    metadata_df.to_csv(metadata_path, index=False)

    tracker = eco2ai.Tracker(
        project_name="GPT2_XL_LoRA_Run", # A distinct project name for this specific experiment
        experiment_description=f"run_{experiment_id}",
        file_name=emissions_path, # Log to the new emissions CSV
        pue=metadata.get('pue', 1.58) # Use PUE from metadata
    )
    tracker.start()

    try:
        # --- Load Tokenizer and Model ---
        model_name = metadata['model_name']
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 tokenizers don't have a pad_token by default; set it to eos_token for classification
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
        
        # Load the model for sequence classification, passing the pad_token_id
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, 
            pad_token_id=tokenizer.pad_token_id 
        )

        # --- LoRA Configuration ---
        lora_config = LoraConfig(
            r=8,                     # LoRA attention dimension (rank)
            lora_alpha=16,           # Scaling factor for LoRA weights
            target_modules=["c_attn", "c_proj", "c_fc"], # Modules to apply LoRA to in GPT-2
            lora_dropout=0.05,       # Dropout probability for LoRA layers
            bias="none",             # Type of bias to add (none, all, lora_only)
            task_type=TaskType.SEQ_CLS # Specify the task for PEFT
        )

        # --- Wrap the base model with LoRA ---
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters() # This will show the small number of trainable parameters

        # --- Dataset Processing ---
        dataset = load_dataset(metadata['dataset_name'], split=f"train[:{metadata['num_train_samples']}]")
        
        def tokenize_function(examples):
            # Tokenize text for classification. Right-padding is generally acceptable.
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=metadata['max_sequence_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # --- Training Arguments ---
        training_args = TrainingArguments(
            output_dir=f"./results/{experiment_id}",
            num_train_epochs=metadata['num_epochs'],
            per_device_train_batch_size=metadata['batch_size'],
            learning_rate=metadata['learning_rate'],
            gradient_accumulation_steps=metadata['gradient_accumulation_steps'],
            fp16=metadata['fp16'], # Enable mixed precision for memory efficiency
            save_strategy="no",    # Don't save checkpoints
            report_to="none",      # Don't report to external tools like W&B
            logging_steps=50,      # Log training metrics more frequently
        )

        # --- Initialize and Run Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            # No data_collator specified, default will be used which is fine for tokenized datasets
        )
        trainer.train()

    except Exception as e:
        print(f"ERROR during experiment: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for easier debugging
    finally:
        tracker.stop() # Ensure tracker stops even if an error occurs
        print(f"--- GPT2-XL LoRA Experiment Finished ---")

if __name__ == '__main__':
    # --- Define the single GPT2-XL LoRA Experiment ---
    experiment_params = {
        'model_name': 'gpt2-xl',
        'dataset_name': 'imdb',
        'num_train_samples': 25000,         # Large subset to ensure significant emissions
        'num_epochs': 5,                    # Substantial epochs
        'batch_size': 8,                    # Small batch size for 1.5B model on P100/T4
        'fp16': True,                       # Mixed precision is crucial
        'pue': 1.58,                        # Representative PUE value
        'learning_rate': 2e-4,              # Common learning rate for LoRA
        'max_sequence_length': 512,         # Max sequence length
        'gradient_accumulation_steps': 2,   # Effective batch size of 16 (8*2)
        'dataset_config': None,             # IMDb has no specific config
    }
    
    # --- Create data/raw directory if it doesn't exist (for eco2ai's emissions.csv) ---
    os.makedirs('data/raw', exist_ok=True) 

    run_gpt2_xl_lora_experiment(experiment_params)