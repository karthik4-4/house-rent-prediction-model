import json
import joblib
import os
from pathlib import Path

metrics_path = Path(
    "c:/house-rent-prediction-model/artifacts/model_evaluation/metrics.json"
)

with open(metrics_path) as f:
    metrics = json.load(f)

best_model = max(
    metrics.items(),
    key=lambda item: item[1]["r2"]
)

model_name = best_model[0]
model_path = best_model[1]['path']

class Predict:
    def __init__(self):
        self.model = joblib.load(Path(model_path))

    def predict(self,data):
        return self.model.predict(data)