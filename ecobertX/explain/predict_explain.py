import joblib
import shap
import pandas as pd
import os
import numpy as np


class PredictionExplainer:

    def __init__(self, project_root):

        self.models_path = os.path.join(project_root, "models")

        # Load existing artifacts from Predictor+
        self.model = joblib.load(
            os.path.join(self.models_path, "best_model.joblib")
        )

        self.preprocessor = joblib.load(
            os.path.join(self.models_path, "preprocessor.joblib")
        )

        # SHAP explainer for tree model (ExtraTrees)
        self.explainer = shap.TreeExplainer(self.model)

        # Load target transform params
        self.y_mean = np.load(os.path.join(self.models_path, "target_mean.npy"))
        self.y_std = np.load(os.path.join(self.models_path, "target_std.npy"))

    # -------------------------------------------------
    # CO2 PREDICTION (same logic as your pipeline)
    # -------------------------------------------------
    def predict(self, raw_row):

        X = self.preprocessor.transform(pd.DataFrame([raw_row]))
        pred_norm = self.model.predict(X)[0]

        # invert target normalization
        pred_log = pred_norm * self.y_std + self.y_mean
        pred = np.expm1(pred_log)

        return float(max(pred, 0))

    # -------------------------------------------------
    # LOCAL SHAP EXPLANATION
    # -------------------------------------------------
    def explain_prediction(self, raw_row):

        X = self.preprocessor.transform(pd.DataFrame([raw_row]))

        shap_vals = self.explainer.shap_values(X)[0]

        features = raw_row.index

        explanation = []

        for f, v in zip(features, shap_vals):
            explanation.append({
                "feature": f,
                "impact": float(v)
            })

        return explanation

    # -------------------------------------------------
    # GLOBAL EXPLANATION PLOT
    # -------------------------------------------------
    def global_explain(self, X_raw):

        X = self.preprocessor.transform(X_raw)
        shap_vals = self.explainer.shap_values(X)

        shap.summary_plot(shap_vals, X_raw)
