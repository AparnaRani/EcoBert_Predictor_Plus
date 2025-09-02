# analyze_model.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay

def analyze_model():
    """
    Loads the trained model and preprocessor to generate and save
    feature importance and partial dependence plots.
    """
    print("Loading model and data...")
    # Load the saved model, preprocessor, and the raw training data
    model = joblib.load('models/emission_predictor_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    X_train_raw = pd.read_csv('data/processed/X_train_raw.csv')
    
    # --- 1. FEATURE IMPORTANCE ---
    print("Calculating feature importance...")
    
    # Get feature names from the preprocessor
    # For categorical features, this gets the one-hot encoded column names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    # For numerical features, it's just the column names
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    
    # Combine them in the correct order
    all_feature_names = np.concatenate([num_features, cat_features])
    
    # Get importance scores from the trained model
    importances = model.feature_importances_
    
    # Create a pandas Series for easy plotting
    forest_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    forest_importances.head(15).plot.bar(ax=ax) # Plot top 15 features
    ax.set_title("Feature importances for CO2 Prediction")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('feature_importance.png') # Save the plot as an image file
    print("Saved feature_importance.png")
    
    # --- 2. PARTIAL DEPENDENCE PLOTS (PDP) ---
    print("Calculating Partial Dependence Plots...")
    
    # We need to use the preprocessed data for PDP
    X_train_processed = preprocessor.transform(X_train_raw)

    # Example 1: PDP for a numerical feature ('num_train_samples')
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_train_processed,
        features=[X_train_raw.columns.get_loc('num_train_samples')], # Use column index
        feature_names=all_feature_names.tolist(),
        ax=ax
    )
    ax.set_title("Partial Dependence of CO2 Emissions on Number of Training Samples")
    plt.tight_layout()
    plt.savefig('pdp_num_samples.png')
    print("Saved pdp_num_samples.png")

    # Example 2: PDP for a categorical feature ('gpu_type')
    # Find the indices of the one-hot encoded 'gpu_type' columns
    gpu_feature_indices = [i for i, col in enumerate(all_feature_names) if col.startswith('gpu_type')]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_train_processed,
        features=gpu_feature_indices,
        feature_names=all_feature_names.tolist(),
        ax=ax
    )
    # Clean up the x-tick labels
    ax.set_xticklabels([label.get_text().replace('gpu_type_', '') for label in ax.get_xticklabels()], rotation=45, ha='right')
    ax.set_title("Partial Dependence of CO2 Emissions on GPU Type")
    plt.tight_layout()
    plt.savefig('pdp_gpu_type.png')
    print("Saved pdp_gpu_type.png")

if __name__ == '__main__':
    analyze_model()