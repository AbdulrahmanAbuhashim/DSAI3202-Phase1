from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

train = pd.read_parquet("data/model_data/train.parquet")
validation = pd.read_parquet("data/model_data/validation.parquet")

numeric = [
    "hour", "month", "is_weekend", "is_night",
    "speed_limit", "high_speed_road"
]

categorical = [
    "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions",
    "urban_or_rural_area", "junction_detail",
    "first_road_class", "junction_control",
    "pedestrian_crossing", "special_conditions_at_site",
    "day_of_week", "second_road_class", "trunk_road_flag",
    "pedestrian_crossing_physical_facilities_historic"
]

features = numeric + categorical

X_train = train[features].copy()
X_val = validation[features].copy()

for col in categorical:
    X_train[col] = X_train[col].astype("string").fillna("missing").astype(str)
    X_val[col] = X_val[col].astype("string").fillna("missing").astype(str)

y_train = train["collision_severity"]
y_val = validation["collision_severity"]

preprocessing = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ), categorical)
])

model = make_pipeline(
    preprocessing,
    ExtraTreesClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        max_features=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

print("Training Extra Trees...", flush=True)
model.fit(X_train, y_train)

predictions = model.predict(X_val)

print("Validation accuracy:", round(
    accuracy_score(y_val, predictions), 4
))
print("Validation macro F1:", round(
    f1_score(y_val, predictions, average="macro"), 4
))

for label, name in [(1, "Fatal"), (2, "Serious"), (3, "Slight")]:
    print(name, "F1:", round(
        f1_score(y_val, predictions, labels=[label],
                 average="macro", zero_division=0), 4
    ))
    print(name, "Recall:", round(
        recall_score(y_val, predictions, labels=[label],
                     average="macro", zero_division=0), 4
    ))

import joblib

Path("outputs/models").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "outputs/models/extra_trees.joblib")
print("Extra Trees model saved")
print("EXTRA TREES EXPERIMENT COMPLETED")
