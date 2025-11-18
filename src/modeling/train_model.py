import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate():
    logger.info("Starting model training and evaluation...")

    project_root = r'D:\EcoPredictor+'
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    models_path = os.path.join(project_root, 'models')

    os.makedirs(models_path, exist_ok=True)

    try:
        preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
        X_train_raw = pd.read_csv(os.path.join(processed_data_path, 'X_train_raw.csv'))
        X_test_raw = pd.read_csv(os.path.join(processed_data_path, 'X_test_raw.csv'))

        y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train_transformed.csv')).values.ravel()
        y_test_transformed = pd.read_csv(os.path.join(processed_data_path, 'y_test_transformed.csv')).values.ravel()
        y_test_original = pd.read_csv(os.path.join(processed_data_path, 'y_test_original.csv')).values.ravel()

        logger.info("Preprocessor and data splits loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading training files: {e}")
        return

    logger.info("Applying preprocessing to train/test sets...")
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    logger.info(f"Preprocessed shapes — X_train: {X_train.shape}, X_test: {X_test.shape}")

    # IMPORTANT: Group by model_name to enforce real generalization
    groups = X_train_raw['model_name'].values
    cv = GroupKFold(n_splits=4)

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Models to evaluate (MLP REMOVED)
    models_to_evaluate = {
        'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoostRegressor': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror', eval_metric='mae'),
        'LGBMRegressor': LGBMRegressor(random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'HuberRegressor': HuberRegressor(max_iter=1000),
        'RidgeRegressor': Ridge(alpha=1.0)
    }

    # Updated + Regularized Param Grids
    param_grids = {
        'RandomForestRegressor': {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'XGBoostRegressor': {
            'n_estimators': [200, 500],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LGBMRegressor': {
            'n_estimators': [200, 500],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [0.1, 1.0],
            'min_child_samples': [10, 20],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3],
            'min_samples_leaf': [2, 4]
        }
    }

    best_model = None
    best_score = float('inf')
    best_model_name = ""

    for name, model in models_to_evaluate.items():
        logger.info(f"Evaluating {name}...")
        grid_search = GridSearchCV(
            model,
            param_grids.get(name, {}),
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        # Pass groups for GroupKFold
        grid_search.fit(X_train, y_train, groups=groups)

        current_mae = -grid_search.best_score_
        logger.info(f"   {name} - Best MAE (CV on log-scale): {current_mae:.4f}")
        logger.info(f"   {name} - Best Params: {grid_search.best_params_}")

        if current_mae < best_score:
            best_score = current_mae
            best_model = grid_search.best_estimator_
            best_model_name = name

    logger.info(f"\n--- Best Model Selected: {best_model_name} (MAE={best_score:.4f}) ---")

    final_model = best_model
    predictions_norm = final_model.predict(X_test)

    y_mean = np.load(os.path.join(models_path, 'target_mean.npy'))
    y_std = np.load(os.path.join(models_path, 'target_std.npy'))
    predictions_log = predictions_norm * y_std + y_mean
    predictions_original = np.expm1(predictions_log)
    predictions_original[predictions_original < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    logger.info("\n--- TEST Results (Original Scale) ---")
    logger.info(f"Model: {best_model_name}")
    logger.info(f"RMSE: {rmse:.4f} kg CO2 | MAE: {mae:.4f} kg | R²: {r2:.4f}")
    logger.info("---------------------------------")

    joblib.dump(final_model, os.path.join(models_path, 'emission_predictor_model.joblib'))

    # Prediction visualization updated
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test_original, y=predictions_original, alpha=0.6)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--')
    plt.xlabel("Actual CO₂ (kg)")
    plt.ylabel("Predicted CO₂ (kg)")
    plt.title("Actual vs Predicted (Test)")
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.subplot(1, 2, 2)
    residuals = y_test_original - predictions_original
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Residual Distribution (Test)")
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plot_save_path = os.path.join(models_path, 'test_predictions_plot.png')
    plt.savefig(plot_save_path)
    logger.info(f"Plot saved: {plot_save_path}")

if __name__ == "__main__":
    np.random.seed(42)
    train_and_evaluate()
