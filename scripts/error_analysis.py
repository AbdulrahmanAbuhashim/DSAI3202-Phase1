import joblib
import pandas as pd
from sklearn.metrics import f1_score

test = pd.read_parquet("data/model_data/test.parquet")
model = joblib.load("outputs/models/gradient_boosting.joblib")

features = [
    "hour", "month", "is_weekend", "is_night",
    "speed_limit", "high_speed_road", "road_type",
    "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area",
    "junction_detail"
]

X = test[features].copy()

categorical = [
    "road_type", "light_conditions", "weather_conditions",
    "road_surface_conditions", "urban_or_rural_area",
    "junction_detail"
]

for col in categorical:
    X[col] = X[col].astype("string").fillna("missing").astype(str)

test["prediction"] = model.predict(X)

print("ERROR ANALYSIS - GRADIENT-BOOSTED TREES")
print("Total wrong predictions:",
      (test["collision_severity"] != test["prediction"]).sum())

print("\nPerformance by urban/rural area:")
for area, group in test.groupby("urban_or_rural_area"):
    score = f1_score(
        group["collision_severity"],
        group["prediction"],
        average="macro",
        labels=[1, 2, 3],
        zero_division=0
    )
    print(area, "- rows:", len(group),
          "- macro F1:", round(score, 4))

print("\nExample incorrect predictions:")
wrong = test[test["collision_severity"] != test["prediction"]]

print(wrong[
    ["collision_index", "collision_severity",
     "prediction", "speed_limit", "weather_conditions"]
].head(10).to_string(index=False))

print("\nERROR ANALYSIS COMPLETED")
