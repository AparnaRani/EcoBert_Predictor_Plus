from explain.predict_explain import PredictionExplainer
from explain.dual_interpreter import interpret_prediction_dual
from observe.file_logger import save_prediction_log

import pandas as pd

PROJECT_ROOT = r"D:\EcoPredictor+"

X_test = pd.read_csv(r"D:\EcoPredictor+\data\processed\X_test_raw.csv")

xai = PredictionExplainer(PROJECT_ROOT)

row = X_test.iloc[0]

# 1. Prediction
co2 = xai.predict(row)

# 2. Explanation
reasons = xai.explain_prediction(row)

# 3. Save logs as evidence
log_file = save_prediction_log(row.to_dict(), co2, reasons)

print("Log saved at:", log_file)

# 4. Human output
interpret_prediction_dual(row, xai)
