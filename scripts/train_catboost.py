import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Use existing training and validation splits
train = pd.read_parquet("data/model_data/train.parquet")
val = pd.read_parquet("data/model_data/validation.parquet")

numeric = [
    "hour", "month", "is_weekend",
    "is_night", "speed_limit", "high_speed_road"
]

categorical = [
    "road_type",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "junction_detail",
    "first_road_class",
    "junction_control",
    "pedestrian_crossing",
    "special_conditions_at_site"
]

features = numeric + categorical
target = "collision_severity"

X_train = train[features].copy()
X_val = val[features].copy()

y_train = train[target].astype(int)
y_val = val[target].astype(int)

# Numeric missing values: use training medians only
medians = {}

for col in numeric:
    X_train[col] = pd.to_numeric(
        X_train[col], errors="coerce"
    )
    X_val[col] = pd.to_numeric(
        X_val[col], errors="coerce"
    )

    median = X_train[col].median()
    if pd.isna(median):
        median = 0

    medians[col] = float(median)
    X_train[col] = X_train[col].fillna(median)
    X_val[col] = X_val[col].fillna(median)

# CatBoost handles categorical features directly
for col in categorical:
    X_train[col] = (
        X_train[col].astype("string")
        .fillna("missing").astype(str)
    )
    X_val[col] = (
        X_val[col].astype("string")
        .fillna("missing").astype(str)
    )

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Features:", len(features), flush=True)

# First experiment: moderate minority-class weights
model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    class_weights={1: 3, 2: 2, 3: 1},
    l2_leaf_reg=5,
    random_seed=42,
    thread_count=-1,
    verbose=100,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical,
    eval_set=(X_val, y_val),
    early_stopping_rounds=70,
    use_best_model=True
)

predictions = np.asarray(
    model.predict(X_val)
).ravel().astype(int)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(
    y_val, predictions, average="macro"
)

print("\n===== CATBOOST RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val, predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_val, predictions, labels=[1, 2, 3]
))

print("\nPrevious Gradient Boosting: 0.3285")
print("Previous Extra Trees: 0.3660")
print("Target: Above 0.5000")

output = Path("outputs/models")
output.mkdir(parents=True, exist_ok=True)

# Save separately; do not overwrite existing models
model.save_model(str(output / "catboost_experiment_1.cbm"))

evidence = Path("evidence/phase2/training")
evidence.mkdir(parents=True, exist_ok=True)

results = {
    "model": "CatBoost",
    "experiment": 1,
    "data_version": "model_data_v1",
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "features": features,
    "numeric_medians": medians,
    "class_weights": {"1": 3, "2": 2, "3": 1},
    "random_seed": 42,
    "best_iteration": model.best_iteration_,
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
    "evaluation": "validation only"
}

(evidence / "catboost_experiment_1.json").write_text(
    json.dumps(results, indent=2)
)

print("\nEXPERIMENT COMPLETED")
