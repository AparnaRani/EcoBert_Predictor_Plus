# File: D:\EcoPredictor+\src\data\make_dataset.py

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_dataset():
    logger.info("Starting data cleaning and merging process...")

    # --- ABSOLUTE PATH DEFINITIONS ---
    # IMPORTANT: Set this to the actual root path of your 'EcoPredictor+' project
    project_root = r'D:\EcoPredictor+' 
    
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    
    # Ensure raw data directory exists
    os.makedirs(raw_data_path, exist_ok=True)
    # --- END ABSOLUTE PATH DEFINITIONS ---

    try:
        # Load the raw datasets from the project root's 'data/raw' folder
        metadata_df = pd.read_csv(os.path.join(raw_data_path, "training_metadata.csv"))
        emissions_df = pd.read_csv(os.path.join(raw_data_path, "emissions.csv"))
        logger.info("Raw 'training_metadata.csv' and 'emissions.csv' loaded.")

        # --- Data Cleaning and Merging Logic ---
        
        # 1. Aggregate emissions: Get the max (final) CO2 emission for each experiment.
        if 'experiment_description' in emissions_df.columns and 'CO2_emissions(kg)' in emissions_df.columns:
            emissions_agg = emissions_df.groupby('experiment_description')['CO2_emissions(kg)'].max().reset_index()
            
            # Extract experiment_id from 'run_...'
            emissions_agg['experiment_id'] = emissions_agg['experiment_description'].str.replace('run_', '')
            logger.info("Emissions aggregated and experiment_id extracted.")
            
            # 2. Merge with metadata
            df = pd.merge(metadata_df, emissions_agg[['experiment_id', 'CO2_emissions(kg)']], on='experiment_id', how='left')
            logger.info(f"Metadata and emissions merged. Initial total samples: {len(df)}")
            
            # Drop rows where the merge failed (e.g., metadata entry without an emissions entry)
            df = df.dropna(subset=['CO2_emissions(kg)'])
            logger.info(f"Dropped rows with missing 'CO2_emissions(kg)'. Remaining samples: {len(df)}")

            # 3. Handle missing 'fp16' values
            df['fp16'] = df['fp16'].fillna(False)
            logger.info(f"Filled missing 'fp16' values with False. Missing 'fp16' values remaining: {df['fp16'].isnull().sum()}")

            # 4. Drop the 'dataset_config' column
            if 'dataset_config' in df.columns:
                df = df.drop(columns=['dataset_config'])
                logger.info("'dataset_config' column dropped.")

            # 5. Drop any other remaining rows with NaN values if any
            initial_rows = len(df)
            df = df.dropna()
            if len(df) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df)} rows with other missing values.")
            
            logger.info(f"Final cleaned and merged dataset size: {len(df)} samples.")

            # Save the cleaned, merged dataset to the project root's 'data/raw' folder
            output_path = os.path.join(raw_data_path, "cleaned_merged_data.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Cleaned and merged data saved to '{output_path}'.")

        else:
            logger.error("Error: 'experiment_description' or 'CO2_emissions(kg)' not found in emissions.csv. Please check file integrity.")

    except FileNotFoundError as e:
        logger.error(f"Error loading raw data files: {e}. Please ensure 'training_metadata.csv' and 'emissions.csv' are in {raw_data_path}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}")

if __name__ == "__main__":
    make_dataset()