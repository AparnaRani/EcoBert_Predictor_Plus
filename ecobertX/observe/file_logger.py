import json
import os
from datetime import datetime
from observe.telemetry_Setup import tracer

LOG_DIR = r"D:\EcoPredictor+\ecobertX\logs"
os.makedirs(LOG_DIR, exist_ok=True)


def save_prediction_log(config, prediction, explanation):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    record = {
        "timestamp": timestamp,
        "input_config": config,
        "predicted_co2": prediction,
        "shap_explanation": explanation
    }

    # ---- Save as JSON file (evidence) ----
    file_path = os.path.join(LOG_DIR, f"log_{timestamp}.json")

    with open(file_path, "w") as f:
        json.dump(record, f, indent=4)

    # ---- Also create OpenTelemetry span ----
    with tracer.start_as_current_span("ecoBERTX_prediction") as span:

        span.set_attribute("co2.prediction", float(prediction))

        for k, v in config.items():
            span.set_attribute(f"input.{k}", str(v))

        for i, r in enumerate(explanation[:5]):
            span.set_attribute(
                f"shap.{i}.{r['feature']}",
                float(r["impact"])
            )

    return file_path
