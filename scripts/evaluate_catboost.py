import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

config = json.loads(
    Path("outputs/models/catboost_config.json").read_text()
)

model = CatBoostClassifier()
model.load_model("outputs/models/catboost_best_checkpoint.cbm")

data = pd.read_parquet("data/model_data/validation.parquet")
X = data[config["feature_order"]].copy()

for col in config["categorical_features"]:
    X[col] = (
        X[col].astype("string")
        .fillna("missing").astype(str)
    )

for col in config["numeric_features"]:
    X[col] = pd.to_numeric(
        X[col], errors="coerce"
    ).fillna(config["numeric_medians"][col])

y = data["collision_severity"].astype(int)
pred = model.predict(X).ravel().astype(int)

labels = [1, 2, 3]

results = {
    "model": "CatBoost",
    "dataset": "validation_2019_2022",
    "data_version": "model_data_v1",
    "rows": len(data),
    "trees": model.tree_count_,
    "accuracy": accuracy_score(y, pred),
    "macro_f1": f1_score(y, pred, average="macro"),
    "classification_report": classification_report(
        y, pred, labels=labels,
        target_names=["Fatal", "Serious", "Slight"],
        output_dict=True, zero_division=0
    )
}

output = Path("evidence/phase2/validation")
output.mkdir(parents=True, exist_ok=True)

(output / "catboost_validation_metrics.json").write_text(
    json.dumps(results, indent=2)
)

matrix = confusion_matrix(y, pred, labels=labels)

pd.DataFrame(
    matrix,
    index=["Actual Fatal", "Actual Serious", "Actual Slight"],
    columns=["Predicted Fatal", "Predicted Serious", "Predicted Slight"]
).to_csv(output / "catboost_confusion.csv")

print("Accuracy:", round(results["accuracy"], 4))
print("Macro F1:", round(results["macro_f1"], 4))
print("Trees:", model.tree_count_)
print("Validation evidence saved.")
