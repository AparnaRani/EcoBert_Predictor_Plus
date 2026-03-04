import joblib
import shap
import pandas as pd
import numpy as np
import os


class PredictionExplainer:

    def __init__(self, project_root):

        print("EcoBERT-X Prediction Explainer initialized.")

        self.models_path = os.path.join(project_root, "models")

        # Load trained model
        self.model = joblib.load(
            os.path.join(self.models_path, "best_model.joblib")
        )

        # Load preprocessor
        self.preprocessor = joblib.load(
            os.path.join(self.models_path, "preprocessor.joblib")
        )

        # Load training data sample as SHAP background
        try:
            background = joblib.load(
                os.path.join(self.models_path, "X_background.joblib")
            )
        except:
            background = None

        # Create SHAP explainer safely
        if background is not None:
            background_transformed = self.preprocessor.transform(background)
            self.explainer = shap.TreeExplainer(
                self.model,
                data=background_transformed,
                feature_perturbation="interventional"
            )
        else:
            self.explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="auto"
            )

        # Load target normalization
        self.y_mean = np.load(
            os.path.join(self.models_path, "target_mean.npy")
        )

        self.y_std = np.load(
            os.path.join(self.models_path, "target_std.npy")
        )


    # ---------------------------------------
    # CLEAN INPUT
    # ---------------------------------------

    def clean_row(self, raw_row):

        row = raw_row.copy()

        row = row.drop(
            labels=[
                c for c in row.index
                if "Unnamed" in str(c)
            ],
            errors="ignore"
        )

        return row


    # ---------------------------------------
    # SAFE SHAP VALUES
    # ---------------------------------------

    def get_shap_values(self, X):

        return self.explainer.shap_values(
            X,
            check_additivity=False
        )[0]


    # ---------------------------------------
    # PREDICT CO2
    # ---------------------------------------

    def predict(self, raw_row):

        row = self.clean_row(raw_row)

        X = self.preprocessor.transform(
            pd.DataFrame([row])
        )

        pred_norm = self.model.predict(X)[0]

        pred_log = pred_norm * self.y_std + self.y_mean

        pred = np.expm1(pred_log)

        return float(max(pred, 0))


    # ---------------------------------------
    # DETAILED EXPLANATION
    # ---------------------------------------

    def explain_prediction_detailed(self, raw_row):

        row = self.clean_row(raw_row)

        X = self.preprocessor.transform(
            pd.DataFrame([row])
        )

        shap_vals = self.get_shap_values(X)

        explanation = []

        for feature, shap_val in zip(row.index, shap_vals):

            val = row[feature]

            # Handle numeric and categorical safely
            try:
                val = float(val)
            except:
                val = str(val)

            explanation.append({

                "feature": feature,

                "value": val,

                "impact": float(shap_val),

                "effect":
                    "increase"
                    if shap_val > 0
                    else "decrease",

                "importance": abs(float(shap_val))

            })

        explanation.sort(
            key=lambda x: x["importance"],
            reverse=True
        )

        return explanation


    # ---------------------------------------
    # CONFIDENCE SCORE
    # ---------------------------------------

    def confidence_score(self, raw_row):

        row = self.clean_row(raw_row)

        X = self.preprocessor.transform(
            pd.DataFrame([row])
        )

        shap_vals = self.get_shap_values(X)

        score = np.mean(np.abs(shap_vals))

        confidence = score / (score + 1)

        return float(confidence)


    # ---------------------------------------
    # MECHANISTIC TRACE
    # ---------------------------------------

    def mechanistic_trace(self, raw_row):

        explanation = self.explain_prediction_detailed(raw_row)

        trace = []

        cumulative = 0

        for e in explanation:

            cumulative += e["impact"]

            trace.append({

                "feature": e["feature"],

                "value": e["value"],

                "impact": e["impact"],

                "cumulative_effect": cumulative

            })

        return trace
