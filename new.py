import pandas as pd

train_df = pd.read_csv(r"D:\EcoPredictor+\data\raw\cleaned_merged_data.csv")
val_df = pd.read_csv(r"D:\EcoPredictor+\data\validation\validation_emissions.csv")

print("=== Training CO2 Range ===")
print(train_df["CO2_emissions(kg)"].describe())
print("\n=== Validation CO2 Range ===")
print(val_df["CO2_emissions(kg)"].describe())
