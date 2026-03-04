import json
import os
from datetime import datetime


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


def save_prediction_log(config, prediction, explanation, confidence, trace):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    record = {

        "timestamp": timestamp,

        "input": config,

        "prediction": prediction,

        "confidence": confidence,

        "explanation": explanation,

        "mechanistic_trace": trace

    }

    path = os.path.join(

        LOG_DIR,

        f"log_{timestamp}.json"

    )

    with open(path, "w") as f:

        json.dump(record, f, indent=4)

    print("Log saved:", path)

    return path
