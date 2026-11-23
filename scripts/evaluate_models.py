import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load the untouched test dataset
test = pd.read_parquet("data/model_data/test.parquet")

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

X_test = test[features].copy()
y_test = test["collision_severity"]

for col in categorical:
    X_test[col] = (
        X_test[col].astype("string")
        .fillna("missing").astype(str)
    )

output = Path("evidence/phase2/validation")
output.mkdir(parents=True, exist_ok=True)

results = {}

models = {
    "Logistic Regression": "outputs/models/logistic_regression.joblib",
    "Gradient-Boosted Trees": "outputs/models/gradient_boosting.joblib"
}

for name, path in models.items():
    model = joblib.load(path)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4)
    }

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    matrix = confusion_matrix(
        y_test, predictions, labels=[1, 2, 3]
    )

    print("Confusion matrix:")
    print(matrix)

    # Save confusion matrix as CSV
    filename = name.lower().replace(" ", "_").replace("-", "_")
    pd.DataFrame(
        matrix,
        index=["Actual Fatal", "Actual Serious", "Actual Slight"],
        columns=["Predicted Fatal", "Predicted Serious", "Predicted Slight"]
    ).to_csv(output / f"{filename}_confusion.csv", index=True)

with open(output / "test_metrics.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nFINAL TEST EVALUATION COMPLETED")
