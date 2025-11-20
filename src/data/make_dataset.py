import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def make_dataset():
    logger.info("Starting data cleaning and merging...")

    project_root = r"D:\EcoPredictor+"
    raw_data_path = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    try:
        meta_path = os.path.join(raw_data_path, "training_metadata.csv")
        emis_path = os.path.join(raw_data_path, "emissions.csv")

        metadata = pd.read_csv(meta_path)
        emissions = pd.read_csv(emis_path)

        logger.info(f"Loaded metadata: {metadata.shape}, emissions: {emissions.shape}")

        # Use final CO2 per experiment_description (max epoch)
        if "experiment_description" not in emissions.columns or "CO2_emissions(kg)" not in emissions.columns:
            logger.error("emissions.csv missing 'experiment_description' or 'CO2_emissions(kg)'")
            return

        emis_agg = (
            emissions.groupby("experiment_description")["CO2_emissions(kg)"]
            .max()
            .reset_index()
        )

        # Your experiment_id seems to be like "run_xxx" in experiment_description
        emis_agg["experiment_id"] = emis_agg["experiment_description"].str.replace("run_", "", regex=False)

        df = metadata.merge(
            emis_agg[["experiment_id", "CO2_emissions(kg)"]],
            on="experiment_id",
            how="inner"
        )
        logger.info(f"Merged dataset: {df.shape}")

        # Handle missing fp16
        if "fp16" in df.columns:
            df["fp16"] = df["fp16"].fillna(False)
        else:
            df["fp16"] = False

        # Drop dataset_config if present
        if "dataset_config" in df.columns:
            df = df.drop(columns=["dataset_config"])

        # Drop remaining NaN rows
        before = len(df)
        df = df.dropna()
        logger.info(f"Dropped {before - len(df)} rows with NaNs. Final: {df.shape}")

        out_path = os.path.join(raw_data_path, "cleaned_merged_data.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved cleaned data to {out_path}")

    except Exception as e:
        logger.error(f"Error during dataset creation: {e}")

if __name__ == "__main__":
    make_dataset()
