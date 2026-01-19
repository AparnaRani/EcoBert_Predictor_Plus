import pandas as pd

df = pd.read_csv(r"D:\EcoPredictor+\data\raw\cleaned_merged_data.csv")

models = sorted(df["model_name"].unique())

print(f"Total unique models: {len(models)}\n")
for m in models:
    print(m)