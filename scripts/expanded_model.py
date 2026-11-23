import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Load training and validation data only
train = pd.read_parquet("data/model_data/train.parquet")
val = pd.read_parquet("data/model_data/validation.parquet")

numeric = [
    "hour", "month", "is_weekend",
    "is_night", "speed_limit", "high_speed_road"
]

categorical = [
    "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions",
    "urban_or_rural_area", "junction_detail",
    "first_road_class", "junction_control",
    "pedestrian_crossing", "special_conditions_at_site",
    "day_of_week", "second_road_class",
    "trunk_road_flag",
    "pedestrian_crossing_physical_facilities_historic"
]

features = numeric + categorical

X_train = train[features].copy()
X_val = val[features].copy()

for col in categorical:
    X_train[col] = X_train[col].astype("string").fillna("missing").astype(str)
    X_val[col] = X_val[col].astype("string").fillna("missing").astype(str)

y_train = train["collision_severity"]
y_val = val["collision_severity"]

# Test three simple configurations
experiments = [
    ("balanced", 1.0, 31),
    ("moderate_weights", 0.5, 31),
    ("smaller_trees", 0.5, 15)
]

best_score = 0
best_name = ""

output = Path("outputs/models")
output.mkdir(parents=True, exist_ok=True)

for name, weight_power, leaves in experiments:
    preprocessing = ColumnTransformer([
        ("numbers", SimpleImputer(strategy="median"), numeric),
        ("categories", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical)
    ])

    model = make_pipeline(
        preprocessing,
        HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=leaves,
            random_state=42
        )
    )

    weights = compute_sample_weight("balanced", y_train)
    weights = weights ** weight_power

    model.fit(
        X_train, y_train,
        histgradientboostingclassifier__sample_weight=weights
    )

    predictions = model.predict(X_val)
    score = f1_score(y_val, predictions, average="macro")

    print("\nExperiment:", name, flush=True)
    print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
    print("Macro F1:", round(score, 4))
    print("Fatal F1:", round(f1_score(
        y_val, predictions, labels=[1],
        average="macro", zero_division=0
    ), 4))
    print("Serious F1:", round(f1_score(
        y_val, predictions, labels=[2],
        average="macro", zero_division=0
    ), 4))

    if score > best_score:
        best_score = score
        best_name = name
        joblib.dump(model, output / "expanded_candidate.joblib")

print("\nBest experiment:", best_name)
print("Best validation macro F1:", round(best_score, 4))
print("Previous candidate macro F1: 0.3071")
print("Improvement:", round(best_score - 0.3071, 4))
print("EXPANDED FEATURE EXPERIMENT COMPLETED")
