import pandas as pd

root = r"D:\EcoPredictor+\data\processed"

X_train = pd.read_csv(root + "/X_train_raw.csv").reset_index(drop=True)
y_train_orig = pd.read_csv(root + "/y_test_original.csv")['y'].reset_index(drop=True)

# restrict to only rows that appear in test set and are "large"
test_large = X_train[X_train['size_cluster']=='large']
idxs = test_large.index

print("\n=== Large Model Test Samples ===")
out = test_large[['model_name','model_parameters','gpu_type',
                  'num_train_samples','num_epochs','batch_size']].copy()
out['CO2_emissions(kg)'] = y_train_orig.loc[idxs].values
print(out)
