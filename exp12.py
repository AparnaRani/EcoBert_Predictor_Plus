# run_gpt2_xl_exp_15.py (GPT2-XL for MAXIMIZED HIGH EMISSIONS within P100/16GB constraints - Variation 11)
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
    Runs a single experiment fine-tuning GPT2-XL with LoRA based on provided params.
    Each call generates unique metadata and emissions files for that run.
    """
    experiment_id = str(uuid.uuid4())

    metadata = params.copy()
    metadata['experiment_id'] = experiment_id

    # Get GPU info if available
    metadata['gpu_type'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    metadata['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # --- CRITICAL: Set model_parameters for gpt2-xl explicitly ---
    metadata['model_parameters'] = 1500000000 # 1.5 Billion parameters for gpt2-xl

    print(f"\n--- Starting GPT2-XL LoRA Experiment ID: {experiment_id} ---")
    print(f"Parameters: {metadata}")

    # --- Define UNIQUE output files for THIS single experiment ---
    output_folder = "gpt2_xl_simultaneous_runs" # Dedicated folder for simultaneous runs
    os.makedirs(output_folder, exist_ok=True) # Ensure the output directory exists

    metadata_path = os.path.join(output_folder, f'metadata_{experiment_id}.csv')
    emissions_path = os.path.join(output_folder, f'emissions_{experiment_id}.csv')

    # Define all expected columns to ensure consistency in the output CSV
    all_columns = [
        'model_name', 'dataset_name', 'num_train_samples', 'num_epochs', 'batch_size',
        'fp16', 'pue', 'experiment_id', 'gpu_type', 'learning_rate', 'max_sequence_length',
        'gradient_accumulation_steps', 'num_gpus', 'dataset_config', 'model_parameters'
    ]

    # Write metadata to its unique file (always with header, as it's a new file)
    metadata_df = pd.DataFrame([metadata], columns=all_columns)
    metadata_df.to_csv(metadata_path, index=False)

    # Initialize eco2ai tracker for this specific experiment run
    tracker = eco2ai.Tracker(
        project_name=f"GPT2_XL_LoRA_Simul_Run_{experiment_id}", # Unique project name
        experiment_description=f"run_{experiment_id}",
        file_name=emissions_path, # Log to the unique emissions CSV
        pue=metadata.get('pue', 1.58) # Use PUE from metadata, default if not provided
    )
    tracker.start()

    try:
        # --- Load Tokenizer and Model ---
        model_name = metadata['model_name']

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,
            pad_token_id=tokenizer.pad_token_id
        )

        # --- LoRA Configuration ---
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS
        )

        model = get_peft_model(model, lora_config)
        print("LoRA activated:")
        model.print_trainable_parameters()

        # --- Dataset Processing ---
        dataset = load_dataset(metadata['dataset_name'], split=f"train[:{metadata['num_train_samples']}]")

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=metadata['max_sequence_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # --- Training Arguments ---
        training_args = TrainingArguments(
            output_dir=f"./results/{experiment_id}",
            num_train_epochs=metadata['num_epochs'],
            per_device_train_batch_size=metadata['batch_size'],
            learning_rate=metadata['learning_rate'],
            gradient_accumulation_steps=metadata['gradient_accumulation_steps'],
            fp16=metadata['fp16'],
            save_strategy="no", report_to="none", logging_steps=100,
            seed=42, # Added seed for reproducibility
        )

        # --- Initialize and Run Trainer ---
        trainer = Trainer(
            model=model, args=training_args, train_dataset=tokenized_dataset,
        )
        trainer.train()

    except Exception as e:
        print(f"ERROR during experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.stop()
        print(f"--- GPT2-XL LoRA Experiment {experiment_id} Finished ---")

if __name__ == '__main__':
    # --- PARAMETERS FOR THIS SPECIFIC GPT-2 XL RUN (Experiment 15 - Variation 11, Higher Emissions) ---
    # These parameters are chosen to MAXIMIZE emissions by making the run very long,
    # while maintaining compatibility with Kaggle's P100 (16GB) memory.
    current_experiment_params = {
        'model_name': 'gpt2-xl',
        'dataset_name': 'imdb',
        'fp16': True,
        'pue': 1.58,
        'learning_rate': 2e-4,

        # --- ADJUSTED FOR KAGGLE P100 (16GB) - FURTHER OPTIMIZED FOR EMISSIONS ---
        'max_sequence_length': 256, # Staying at 256 for proven memory fit
        'num_train_samples': 25000, # Keep full dataset
        'num_epochs': 78,           # Increased to 78 for very long runtime
        'batch_size': 1,            # Remains at 1, essential for memory
        'gradient_accumulation_steps': 14, # Set to 14 (effective batch size 1*14=14)

        'dataset_config': None, # Keep as is
    }

    # --- Create data/raw directory if it doesn't exist (for eco2ai's internal usage) ---
    os.makedirs('data/raw', exist_ok=True)

    # Run the single, defined experiment
    run_gpt2_xl_lora_experiment(current_experiment_params)

    print("\nGPT2-XL LoRA experiment executed. Check 'gpt2_xl_simultaneous_runs' folder for unique output files.")