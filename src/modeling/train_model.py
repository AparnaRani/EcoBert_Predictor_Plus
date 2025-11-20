import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def invert_transform(pred_norm, y_mean, y_std):
    pred_log = pred_norm * y_std + y_mean
    pred_kg = np.expm1(pred_log)
    pred_kg[pred_kg < 0] = 0
    return pred_kg

def evaluate_group(name, true_vals, pred_vals):
    if len(true_vals) == 0:
        return {"RMSE": None, "MAE": None, "R2": None}

    return {
        "RMSE": np.sqrt(mean_squared_error(true_vals, pred_vals)),
        "MAE": mean_absolute_error(true_vals, pred_vals),
        "R2": r2_score(true_vals, pred_vals)
    }

def train_and_evaluate():
    logger.info("=== Evaluate All Models ===")

    project_root = r"D:\EcoPredictor+"
    processed_data_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")
    os.makedirs(models_path, exist_ok=True)

    preprocessor = joblib.load(os.path.join(models_path, "preprocessor.joblib"))

    X_train_raw = pd.read_csv(os.path.join(processed_data_path, "X_train_raw.csv"))
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, "X_test_raw.csv"))

    y_train = pd.read_csv(os.path.join(processed_data_path, "y_train_transformed.csv"))["y"].values
    y_test_original = pd.read_csv(os.path.join(processed_data_path, "y_test_original.csv"))["y"].values

    # Transform features
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_mean = np.load(os.path.join(models_path, "target_mean.npy"))
    y_std = np.load(os.path.join(models_path, "target_std.npy"))

    # Group masks
    test_small = (X_test_raw["size_cluster"] == "small")
    test_large = (X_test_raw["size_cluster"] == "large")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            depth=6, iterations=500, learning_rate=0.05,
            loss_function="RMSE", verbose=False, random_seed=42
        ),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        pred_norm = model.predict(X_test)
        pred_kg = invert_transform(pred_norm, y_mean, y_std)

        results[name] = {
            "Overall": evaluate_group(name, y_test_original, pred_kg),
            "SmallModels": evaluate_group(name, y_test_original[test_small], pred_kg[test_small]),
            "LargeModels": evaluate_group(name, y_test_original[test_large], pred_kg[test_large]),
        }

    # Save results table
    results_df = pd.DataFrame({
        (m + "_" + k): pd.Series(v[k])
        for m, v in results.items()
        for k in v.keys()
    }).T

    results_csv = os.path.join(models_path, "model_comparison_metrics.csv")
    results_df.to_csv(results_csv)
    logger.info(f"\nResults saved → {results_csv}")

    # Print beautifully
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\n===== MODEL PERFORMANCE RESULTS =====\n")
        print(results_df)
        print("\n=====================================\n")

if __name__ == "__main__":
    train_and_evaluate()
