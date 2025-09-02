# merge_data.py
import pandas as pd
import os

def consolidate_metadata():
    """
    Merges all partial metadata CSVs from a source folder into one clean, 
    final file, filling in any missing columns with sensible default values.
    """
    source_folder = 'temp_data'
    final_output_path = 'data/raw/training_metadata.csv'
    
    # This is the complete "master" list of all columns that should exist in the final file.
    all_columns = [
        'model_name', 'dataset_name', 'num_train_samples', 'num_epochs', 
        'batch_size', 'fp16', 'pue', 'experiment_id', 'gpu_type', 
        'learning_rate', 'max_sequence_length', 'gradient_accumulation_steps', 
        'num_gpus', 'dataset_config'
    ]
    
    # These are the default values the script used if a parameter wasn't specified in an experiment.
    default_values = {
        'learning_rate': 2e-5,
        'max_sequence_length': 512,
        'gradient_accumulation_steps': 1,
        'num_gpus': 1,
        'dataset_config': None
    }

    all_dataframes = []
    
    print(f"Reading files from the '{source_folder}' directory...")
    # Loop through every file in your temp_data folder.
    for filename in os.listdir(source_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(source_folder, filename)
            print(f"Processing '{filename}'...")
            
            # Read the current CSV file.
            df = pd.read_csv(filepath)
            
            # Check for any columns from the master list that are missing in this file.
            for col in all_columns:
                if col not in df.columns:
                    # If a column is missing, add it and fill it with its default value.
                    df[col] = default_values.get(col)
            
            # Ensure all columns are in the same, correct order and drop any extra ones.
            df = df.reindex(columns=all_columns)
            all_dataframes.append(df)

    if not all_dataframes:
        print("No CSV files found to merge.")
        return

    # Combine all the cleaned-up dataframes into one big dataframe.
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove any accidental duplicate rows.
    final_df.drop_duplicates(inplace=True)
    
    print(f"\nTotal unique samples collected: {len(final_df)}")
    
    # Save the final, clean file to the correct location.
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    print(f"Successfully saved consolidated data to '{final_output_path}'")

if __name__ == '__main__':
    consolidate_metadata()