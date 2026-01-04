import os
import joblib
import argparse
import numpy as np
import pandas as pd

# Paths
project_root = r"D:\EcoPredictor+"
models_dir = os.path.join(project_root, "models")

# Load Predictors
model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
y_mean = np.load(os.path.join(models_dir, "target_mean.npy"))
y_std = np.load(os.path.join(models_dir, "target_std.npy"))

# CLI Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--params", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--seq", type=int, required=True)
parser.add_argument("--pue", type=float, required=True)
parser.add_argument("--gpu", default="Unknown-GPU")
parser.add_argument("--gpu_watts", type=int, default=250)
args = parser.parse_args()

# Infer derived features
num_train_samples = 10000  # We assume common value
total_tokens = args.batch * args.seq * num_train_samples

compute_load = total_tokens * args.params
compute_log = np.log(max(compute_load, 1))

log_params = np.log(max(args.params, 1))

# Size cluster mapping
if args.params <= 200_000_000:
    size_cluster = "small"
elif args.params <= 1_000_000_000:
    size_cluster = "medium"
else:
    size_cluster = "large"

# Build complete row
row = pd.DataFrame([{
    "model_name": args.model,
    "dataset_name": args.dataset,
    "num_train_samples": num_train_samples,
    "num_epochs": args.epochs,
    "batch_size": args.batch,
    "fp16": True,
    "pue": args.pue,
    "gpu_type": args.gpu,
    "learning_rate": args.lr,
    "max_sequence_length": args.seq,
    "gradient_accumulation_steps": 4,
    "num_gpus": 1,
    "model_parameters": args.params,
    "log_model_parameters": log_params,
    "total_tokens": total_tokens,
    "compute_load": compute_load,
    "compute_log": compute_log,
    "size_cluster": size_cluster,
    "gpu_power_watts": args.gpu_watts,
    "model_family": args.model.split("-")[0],
}])

# Predict
row_proc = preprocessor.transform(row)
pred_norm = model.predict(row_proc)
pred_log = pred_norm * y_std + y_mean
pred_kg = float(np.expm1(pred_log))

print("\n====================================")
print("📌 PREDICTION RESULT")
print("====================================")
print(f"Model Name      : {args.model}")
print(f"GPU Type        : {args.gpu}")
print(f"Predicted CO₂   : {pred_kg:.9f} kg")
print("====================================\n")
