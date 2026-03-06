import numpy as np

def stability_test(explainer, row, runs=5):

    explanations = []

    for _ in range(runs):
        exp = explainer.explain_prediction_detailed(row)
        explanations.append([r["impact"] for r in exp])

    explanations = np.array(explanations)

    stability_score = np.std(explanations)

    return stability_score