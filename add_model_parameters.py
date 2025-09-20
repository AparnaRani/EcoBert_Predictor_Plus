# add_model_parameters.py
import pandas as pd
import os

def add_parameters_to_metadata(metadata_path='data/raw/training_metadata.csv'):
    """
    Reads the metadata CSV, adds a 'model_parameters' column based on 'model_name',
    and saves the updated CSV.
    """
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    # Define the mapping from model_name to model_parameters
    # Add any other models you've used to this dictionary!
    model_param_map = {
        'prajjwal1/bert-tiny': 4_400_000,
        'prajjwal1/bert-mini': 11_000_000,
        'albert-base-v2': 12_000_000,
        'distilbert-base-uncased': 66_000_000,
        't5-small': 60_000_000, # If you have T5 small data
        'bert-base-uncased': 110_000_000,
        'roberta-base': 125_000_000,
        'bert-large-uncased': 340_000_000,
        'gpt2-xl': 1_500_000_000, # 1.5 Billion (if you add LoRA data later)
        # Add entries for any other models you might have in your dataset
    }

    # Apply the mapping. If a model_name is not found, set parameters to 0 or NaN (or handle error)
    # Using .get() with a default value (e.g., 0) is safer for models not in the map
    df['model_parameters'] = df['model_name'].apply(lambda x: model_param_map.get(x, 0))

    # Check for any models that weren't mapped
    unmapped_models = df[df['model_parameters'] == 0]['model_name'].unique()
    if len(unmapped_models) > 0:
        print(f"\nWarning: The following models had 0 parameters assigned (not found in map):")
        for model in unmapped_models:
            print(f"- {model}")
        print("Please add them to the 'model_param_map' in the script if they are valid models.")

    # Save the updated DataFrame back to the original path, overwriting it
    df.to_csv(metadata_path, index=False)
    print(f"\nSuccessfully updated {metadata_path} with 'model_parameters' column.")
    print(f"Shape of updated metadata: {df.shape}")

if __name__ == "__main__":
    add_parameters_to_metadata()