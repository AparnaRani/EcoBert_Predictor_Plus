from explain.predict_explain import PredictionExplainer
from validation.faithfulness import faithfulness_test
from validation.stability import stability_test
from validation.infidelity import infidelity_test

import pandas as pd

PROJECT_ROOT = r"D:\EcoPredictor+"

xai = PredictionExplainer(PROJECT_ROOT)

X_test = pd.read_csv(
    r"D:\EcoPredictor+\data\processed\X_test_raw.csv"
)

row = X_test.iloc[0]

faith = faithfulness_test(xai, row)
stab = stability_test(xai, row)
infid = infidelity_test(xai, row)

print("\nExplainability Validation Results")
print("----------------------------------")
print("Faithfulness :", faith)
print("Stability :", stab)
print("Infidelity :", infid)