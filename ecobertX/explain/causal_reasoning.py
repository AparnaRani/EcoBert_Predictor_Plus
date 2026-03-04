def generate_causal_reason(feature, value, impact):

    direction = "increased" if impact > 0 else "reduced"

    explanations = {

        "batch_size":
        f"Batch size of {value} {direction} CO₂ because batch size directly controls GPU memory usage and parallel compute load. Smaller batch sizes require fewer GPU operations, reducing energy consumption.",

        "max_sequence_length":
        f"Sequence length of {value} {direction} CO₂ because longer sequences increase transformer attention computation complexity O(n²), increasing GPU energy usage.",

        "log_model_parameters":
        f"Model parameter scale {value} {direction} CO₂ because larger models require more matrix multiplications and GPU compute.",

        "compute_log":
        f"Compute workload level {value} {direction} CO₂ because compute intensity directly correlates with GPU power draw.",

        "dataset_name":
        f"Dataset '{value}' {direction} CO₂ due to differences in training complexity and convergence efficiency.",

        "total_tokens":
        f"Token count {value} {direction} CO₂ because more tokens require more forward and backward passes.",

        "model_name":
        f"Model architecture '{value}' {direction} CO₂ due to differences in computational efficiency.",

        "num_epochs":
        f"Training epochs {value} {direction} CO₂ because more epochs require repeated training cycles.",

        "learning_rate":
        f"Learning rate {value} {direction} CO₂ because inefficient learning rates increase training duration.",

        "fp16":
        f"FP16 precision {direction} CO₂ because lower precision reduces compute and memory load."

    }

    if feature in explanations:
        return explanations[feature]

    return f"{feature} {direction} CO₂ due to its influence on compute workload."
