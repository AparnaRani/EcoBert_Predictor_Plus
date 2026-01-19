import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Invert target normalization
# -------------------------------------------------
def invert_target(pred_norm, y_mean, y_std):
    pred_log = pred_norm * y_std + y_mean
    pred = np.expm1(pred_log)
    pred[pred < 0] = 0
    return pred

# -------------------------------------------------
# Train & select best model
# -------------------------------------------------
def train_and_select_best():
    logger.info("Training models...")

    project_root = r"D:\EcoPredictor+"
    processed_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")

    # -------- Load artifacts --------
    preprocessor = joblib.load(os.path.join(models_path, "preprocessor.joblib"))

    X_train_raw = pd.read_csv(os.path.join(processed_path, "X_train_raw.csv"))
    X_test_raw = pd.read_csv(os.path.join(processed_path, "X_test_raw.csv"))

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_train = pd.read_csv(
        os.path.join(processed_path, "y_train_transformed.csv")
    )["y"].values

    y_test = pd.read_csv(
        os.path.join(processed_path, "y_test_original.csv")
    )["y"].values

    y_mean = np.load(os.path.join(models_path, "target_mean.npy"))
    y_std = np.load(os.path.join(models_path, "target_std.npy"))

    # -------- Models --------
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_seed=42
        ),
    }

    best_model = None
    best_name = None
    best_r2 = -1e9
    results = []

    # -------- Train & evaluate --------
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        preds_norm = model.predict(X_test)
        preds = invert_target(preds_norm, y_mean, y_std)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # ✅ FIX
        mae = mean_absolute_error(y_test, preds)

        results.append([name, r2, rmse, mae])
        logger.info(f"{name} → R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    # -------- Save results --------
    results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAE"])
    results_df.to_csv(
        os.path.join(models_path, "model_performance_summary.csv"),
        index=False
    )

    joblib.dump(best_model, os.path.join(models_path, "best_model.joblib"))

    logger.info("====================================")
    logger.info(f"🏆 Best Model: {best_name} (R2={best_r2:.4f})")
    logger.info("Training complete.")

# -------------------------------------------------
if __name__ == "__main__":
    train_and_select_best()
