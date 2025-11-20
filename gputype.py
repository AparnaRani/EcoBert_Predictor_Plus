import pandas as pd
df = pd.read_csv(r"D:\EcoPredictor+\data\raw\cleaned_merged_data.csv")
print(df['gpu_type'].unique())
