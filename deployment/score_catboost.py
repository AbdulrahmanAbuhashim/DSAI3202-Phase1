import json
import os
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

model = None
config = None


def init():
    global model, config

    root = Path(
        os.environ.get("AZUREML_MODEL_DIR", "deployment/catboost_model")
    )

    models = list(root.rglob("catboost_best_checkpoint.cbm"))
    configs = list(root.rglob("catboost_config.json"))

    if len(models) != 1 or len(configs) != 1:
        raise RuntimeError("CatBoost model or configuration not found")

    config = json.loads(configs[0].read_text())

    model = CatBoostClassifier()
    model.load_model(str(models[0]))

    if model.feature_names_ != config["feature_order"]:
        raise RuntimeError("Model feature order does not match configuration")

    print("CatBoost initialized successfully")


def run(raw_data):
    try:
        if model is None or config is None:
            raise RuntimeError("Model not initialized")

        request = (
            json.loads(raw_data)
            if isinstance(raw_data, (str, bytes))
            else raw_data
        )

        if not isinstance(request, dict):
            raise ValueError("Expected a JSON object")

        records = request.get("input_data")

        if not isinstance(records, list) or not records:
            raise ValueError("input_data must be a non-empty list")

        if not all(isinstance(r, dict) for r in records):
            raise ValueError("Each input record must be an object")

        features = config["feature_order"]

        for i, record in enumerate(records):
            missing = [f for f in features if f not in record]
            if missing:
                raise ValueError(
                    f"Record {i} missing required fields: {missing}"
                )

        X = pd.DataFrame(records)[features].copy()

        for col in config["categorical_features"]:
            X[col] = (
                X[col]
                .astype("string")
                .fillna(config["categorical_missing_value"])
                .astype(str)
            )

        for col in config["numeric_features"]:
            converted = pd.to_numeric(X[col], errors="coerce")
            invalid = X[col].notna() & converted.isna()

            if invalid.any():
                raise ValueError(f"Invalid numeric value for {col}")

            X[col] = converted.fillna(
                config["numeric_medians"][col]
            )

        predictions = (
            model.predict(X).ravel().astype(int).tolist()
        )

        return {
            "predictions": predictions,
            "model_version": "road-accident-severity:3"
        }

    except (ValueError, TypeError, KeyError) as error:
        return {
            "error": str(error),
            "model_version": "road-accident-severity:3"
        }
