import json
import os
from pathlib import Path

import joblib
import pandas as pd

numeric = [
    "hour", "month", "is_weekend",
    "is_night", "speed_limit", "high_speed_road"
]

categorical = [
    "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions",
    "urban_or_rural_area", "junction_detail",
    "first_road_class", "junction_control",
    "pedestrian_crossing", "special_conditions_at_site"
]

features = numeric + categorical
model = None


def init():
    global model
    folder = Path(os.environ["AZUREML_MODEL_DIR"])
    model_file = next(folder.rglob("improved_candidate.joblib"))
    model = joblib.load(model_file)


def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

        if not isinstance(data, dict) or "rows" not in data:
            return {"error": "Request must contain rows"}

        rows = data["rows"]

        if not isinstance(rows, list) or not rows:
            return {"error": "Rows must be a non-empty list"}

        if any(not isinstance(row, dict) for row in rows):
            return {"error": "Each row must be an object"}

        df = pd.DataFrame(rows)
        missing = [col for col in features if col not in df.columns]

        if missing:
            return {"error": "Missing fields: " + ", ".join(missing)}

        df = df[features].copy()

        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="raise")

        for col in categorical:
            df[col] = df[col].astype("string").fillna("missing").astype(str)

        predictions = model.predict(df)

        return {
            "predictions": predictions.tolist(),
            "model_version": "road-accident-severity:1"
        }

    except Exception as error:
        return {"error": str(error)}
