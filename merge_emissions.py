# merge_emissions.py
import pandas as pd
import os

def merge_emissions_data():
    """A simple script to merge only the emissions.csv files."""
    source_folder = 'temp_emissions'
    final_output_path = 'data/raw/emissions.csv'

    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        return

    all_dfs = []
    print(f"Reading all emissions files from '{source_folder}'...")
    for filename in os.listdir(source_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(source_folder, filename)
            all_dfs.append(pd.read_csv(filepath))

    if not all_dfs:
        print("No CSV files found to merge.")
        return

    # Combine all data into one file
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.drop_duplicates(inplace=True)

    # Save the final, merged file
    os.makedirs('data/raw', exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    print(f"Successfully merged all emissions data into '{final_output_path}' ✅")

if __name__ == '__main__':
    merge_emissions_data()