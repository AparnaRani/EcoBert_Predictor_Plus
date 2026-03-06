import numpy as np

def infidelity_test(explainer, row):

    explanation = explainer.explain_prediction_detailed(row)

    impacts = np.array([abs(r["impact"]) for r in explanation])

    prediction = explainer.predict(row)

    noise = np.random.normal(0, 0.01, len(impacts))

    perturbed = impacts + noise

    infidelity = np.mean((perturbed - impacts) ** 2)

    return infidelity