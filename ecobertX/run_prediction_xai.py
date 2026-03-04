from explain.predict_explain import PredictionExplainer
from explain.dual_interpreter import interpret_prediction_full
from visualize.heatmap import plot_heatmap

import pandas as pd

PROJECT_ROOT = r"D:\EcoPredictor+"

X_test = pd.read_csv(
    r"D:\EcoPredictor+\data\processed\X_test_raw.csv"
)

xai = PredictionExplainer(PROJECT_ROOT)

row = X_test.iloc[0]

prediction = xai.predict(row)

explanation = xai.explain_prediction_detailed(row)

interpret_prediction_full(row, xai)

plot_heatmap(
    explanation,
    r"D:\EcoPredictor+\ecobertX\logs\heatmap.png"
)
