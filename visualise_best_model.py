# visualize_best_model.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import os

# Import the models to get the best one from the comparison script
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def create_visualizations():
    """
    Loads the best trained model and generates a suite of
    visualizations to showcase its performance and workings.
    """
    print("Loading model and data...")
    models_path = 'models'
    processed_data_path = 'data/processed'
    
    # --- Load all necessary components ---
    preprocessor = joblib.load(os.path.join(models_path, 'preprocessor.joblib'))
    # Load the raw data splits to get the true values and feature names
    X_train_raw = pd.read_csv(os.path.join(processed_data_path, 'X_train_raw.csv'))
    X_test_raw = pd.read_csv(os.path.join(processed_data_path, 'X_test_raw.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()
    
    # --- Re-create and load the best model pipeline ---
    # NOTE: We are assuming Random Forest was the best, as per your results.
    # If another model was better, you would load that one instead.
    best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', best_model)])
    # The `train_and_compare_models` script would have saved the best model,
    # but for clarity, we can just re-train it here quickly as it's fast.
    print("Re-training the best model (Random Forest) to ensure consistency...")
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    pipeline.fit(X_train_raw, y_train)
    
    # Make predictions on the test set
    predictions = pipeline.predict(X_test_raw)

    # --- 1. Actual vs. Predicted Plot ---
    print("Generating Actual vs. Predicted plot...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel("Actual CO2 Emissions (kg)")
    plt.ylabel("Predicted CO2 Emissions (kg)")
    plt.title("Actual vs. Predicted Emissions")
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved actual_vs_predicted.png")
    plt.close()

    # --- 2. Residuals Plot ---
    print("Generating Residuals plot...")
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted CO2 Emissions (kg)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs. Predicted Values")
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    print("Saved residuals_plot.png")
    plt.close()
    
    # --- 3. Feature Importance Plot ---
    print("Generating Feature Importance plot...")
    # Get feature names from the preprocessor
    cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    num_features = pipeline.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
    all_feature_names = np.concatenate([num_features, cat_features])
    
    importances = pipeline.named_steps['regressor'].feature_importances_
    forest_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    forest_importances.head(15).plot.bar(ax=ax)
    ax.set_title("Top 15 Feature Importances for Random Forest")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")
    plt.close()
    
    # --- 4. SHAP Summary Plot ---
    print("Generating SHAP Summary plot... (This may take a moment)")
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test_raw)
    
    # Create the explainer
    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_processed)
    
    # Plotting
    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=all_feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150) # Use dpi for better resolution
    print("Saved shap_summary.png")
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    create_visualizations()