import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Load training and validation data
train = pd.read_parquet("data/model_data/train.parquet")
val = pd.read_parquet("data/model_data/validation.parquet")

numeric = [
    "hour", "month", "is_weekend",
    "is_night", "speed_limit", "high_speed_road"
]

categorical = [
    "road_type", "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area",
    "junction_detail"
]

features = numeric + categorical

X_train = train[features].copy()
X_val = val[features].copy()

for col in categorical:
    X_train[col] = X_train[col].astype("string").fillna("missing").astype(str)
    X_val[col] = X_val[col].astype("string").fillna("missing").astype(str)

y_train = train["collision_severity"]
y_val = val["collision_severity"]

# Prepare features using training data only
preprocessing = ColumnTransformer([
    ("numbers", SimpleImputer(strategy="median"), numeric),
    ("categories", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical)
])

# Give more weight to less common severity classes
weights = compute_sample_weight("balanced", y_train)

model = make_pipeline(
    preprocessing,
    HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42
    )
)

model.fit(
    X_train, y_train,
    histgradientboostingclassifier__sample_weight=weights
)

predictions = model.predict(X_val)

results = {
    "model": "Balanced Gradient-Boosted Trees",
    "data_version": "features_v1",
    "training_rows": len(train),
    "validation_rows": len(val),
    "random_state": 42,
    "max_iter": 100,
    "accuracy": float(accuracy_score(y_val, predictions)),
    "macro_f1": float(f1_score(y_val, predictions, average="macro"))
}

output = Path("outputs/models")
output.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output / "gradient_boosting.joblib")

with open(output / "candidate_metrics.json", "w") as file:
    json.dump(results, file, indent=2)

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Accuracy:", round(results["accuracy"], 4))
print("Macro F1:", round(results["macro_f1"], 4))
print(classification_report(y_val, predictions, zero_division=0))
print("CANDIDATE TRAINING COMPLETED")
