import pandas as pd

df_best = pd.read_csv(r"D:\EcoPredictor+\data\processed\X_test_raw.csv")
print(df_best["model_name"].unique())
