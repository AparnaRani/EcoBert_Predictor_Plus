import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "experiment_id",
    "model_name",
    "model_parameters",
    "num_train_samples",
    "num_epochs",
    "max_sequence_length",
    "gpu_type",
    "CO2_emissions(kg)",
]

def make_dataset():
    logger.info("Starting data cleaning and merging...")

    project_root = r"D:\EcoPredictor+"
    raw_path = os.path.join(project_root, "data", "raw")

    meta_path = os.path.join(raw_path, "training_metadata.csv")
    emis_path = os.path.join(raw_path, "emissions.csv")

    metadata = pd.read_csv(meta_path)
    emissions = pd.read_csv(emis_path)

    logger.info(f"Loaded metadata: {metadata.shape}")
    logger.info(f"Loaded emissions: {emissions.shape}")

    # ---- Extract experiment_id from emissions if needed ----
    if "experiment_id" not in emissions.columns:
        emissions["experiment_id"] = emissions["experiment_description"].str.extract(
            r"(?:experiment_id=)?([a-f0-9\-]{6,})",
            expand=False
        )

    before = len(emissions)
    emissions = emissions.dropna(subset=["experiment_id"])
    logger.info(f"Dropped {before - len(emissions)} emissions rows without experiment_id")

    # ---- Aggregate CO2 ----
    emis_agg = (
        emissions.groupby("experiment_id")["CO2_emissions(kg)"]
        .max()
        .reset_index()
    )

    # ---- Merge ----
    df = metadata.merge(emis_agg, on="experiment_id", how="inner")
    logger.info(f"Merged dataset shape: {df.shape}")

    # ---- fp16 safety ----
    if "fp16" in df.columns:
        df["fp16"] = df["fp16"].fillna(False)
    else:
        df["fp16"] = False

    # ---- Drop ONLY required NaNs ----
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    logger.info(f"Dropped {before - len(df)} rows missing REQUIRED fields")

    out_path = os.path.join(raw_path, "cleaned_merged_data.csv")
    df.to_csv(out_path, index=False)

    logger.info(f"Saved cleaned dataset → {out_path}")
    logger.info(f"FINAL ROW COUNT = {len(df)}")

if __name__ == "__main__":
    make_dataset()
