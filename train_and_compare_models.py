# train_and_compare_models.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# Import all the models we want to test
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_compare_and_visualize():
    """
    Loads data, trains multiple models, compares their performance,
    and generates visualizations for the best one.
    """
    # --- 1. Load Data ---
    print("Loading preprocessed data...")
    models_path = 'models'
    processed_data_path = 'data/processed'
    
    preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
    X_train_raw = pd.read_csv(os.path.join(processed_data_path, 'X_train_raw.csv'))
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, 'X_test_raw.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()

    # --- 2. Define Models ---
    # A dictionary of all models we want to train
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR()
    }

    results = {}

    # --- 3. Train and Evaluate Each Model ---
    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        # Create a full pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', model)])
        
        # Train the model
        pipeline.fit(X_train_raw, y_train)
        
        # Make predictions
        predictions = pipeline.predict(X_test_raw)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Store results
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model_object': pipeline}
        print(f"Done. R-squared: {r2:.4f}")

    # --- 4. Compare Models and Visualize Performance ---
    results_df = pd.DataFrame(results).T.drop(columns=['model_object'])
    print("\n--- Model Comparison ---")
    print(results_df)

    # Plotting the comparison
    results_df['R2'].sort_values().plot(kind='barh', figsize=(10, 6))
    plt.title("Model Comparison: R-squared (Higher is Better)")
    plt.xlabel("R-squared Score")
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nSaved model_comparison.png")
    plt.close()

    # --- 5. Visualize the BEST Model ---
    # Find the best model based on R-squared score
    best_model_name = results_df['R2'].idxmax()
    best_pipeline = results[best_model_name]['model_object']
    print(f"\nBest performing model is: {best_model_name}")

    # For tree-based models (Random Forest, Gradient Boosting), we can plot feature importance
    if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
        print("Generating feature importance for the best model...")
        
        # Get feature names from the preprocessor
        cat_features = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        num_features = best_pipeline.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
        all_feature_names = np.concatenate([num_features, cat_features])
        
        importances = best_pipeline.named_steps['regressor'].feature_importances_
        forest_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        forest_importances.head(15).plot.bar(ax=ax)
        ax.set_title(f"Feature Importances for {best_model_name}")
        ax.set_ylabel("Importance")
        fig.tight_layout()
        plt.savefig('best_model_feature_importance.png')
        print("Saved best_model_feature_importance.png")
        plt.close()

if __name__ == "__main__":
    # You might need to install seaborn for better plotting styles
    # pip install seaborn
    sns.set_theme(style="whitegrid")
    train_compare_and_visualize()