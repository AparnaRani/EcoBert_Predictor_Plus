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

def invert_target(pred_norm, y_mean, y_std):
    pred_log = pred_norm * y_std + y_mean
    pred_kg = np.expm1(pred_log)
    pred_kg[pred_kg < 0] = 0
    return pred_kg


def train_and_select_best():
    logger.info("=== Training & Model Selection ===")

    project_root = r"D:\EcoPredictor+"
    processed = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    X_train_raw = pd.read_csv(os.path.join(processed, "X_train_raw.csv"))
    X_test_raw = pd.read_csv(os.path.join(processed, "X_test_raw.csv"))

    y_train = pd.read_csv(os.path.join(processed, "y_train_transformed.csv"))["y"].values
    y_test_org = pd.read_csv(os.path.join(processed, "y_test_original.csv"))["y"].values

    # Transform feature inputs
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
    y_std = np.load(os.path.join(models_dir, "target_std.npy"))

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            objective="reg:squarederror", random_state=42),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostRegressor(
            depth=6, iterations=500, learning_rate=0.05,
            loss_function="RMSE", verbose=False, random_seed=42),
    }

    best_model = None
    best_name = None
    best_r2 = -999

    results = []

    # Train + Evaluate all models
    for name, model in models.items():
        logger.info(f"\n⚙ Training model: {name}")
        model.fit(X_train, y_train)

        pred_norm = model.predict(X_test)
        pred_kg = invert_target(pred_norm, y_mean, y_std)

        r2 = r2_score(y_test_org, pred_kg)
        rmse = np.sqrt(mean_squared_error(y_test_org, pred_kg))
        mae = mean_absolute_error(y_test_org, pred_kg)

        results.append([name, r2, rmse, mae])
        print(f"{name} → R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        # Select best using R²
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    # Save results table
    results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAE"])
    results_path = os.path.join(models_dir, "model_performance_summary.csv")
    results_df.to_csv(results_path, index=False)

    print("\n=== PERFORMANCE SUMMARY ===")
    print(results_df)
    print("Saved:", results_path)

    # Save best model
    best_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_path)

    print(f"\n🏆 Best Model = {best_name}  (R2={best_r2:.4f})")
    print(f"📌 Saved as: {best_path}")


if __name__ == "__main__":
    train_and_select_best()
