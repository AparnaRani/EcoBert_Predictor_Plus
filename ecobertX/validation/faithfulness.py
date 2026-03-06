import numpy as np

def faithfulness_test(explainer, row, top_k=3):

    original_pred = explainer.predict(row)

    explanation = explainer.explain_prediction_detailed(row)

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    top_features = explanation[:top_k]

    perturbed = row.copy()

    for f in top_features:

        feature = f["feature"]

        if feature in perturbed:

            if isinstance(perturbed[feature], (int, float)):
                perturbed[feature] *= 2   # stronger perturbation

    new_pred = explainer.predict(perturbed)

    faithfulness = abs(original_pred - new_pred)

    return faithfulness