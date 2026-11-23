from pathlib import Path
import pandas as pd

SOURCE = Path("data/feature_training/projects/road-accident-severity/features_v1")
OUTPUT = Path("data/model_data")
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("part_*.parquet"))
assert len(files) == 183, "Expected 183 feature files"

train = []
validation = []
test = []

for file in files:
    df = pd.read_parquet(file)

    # Select about 5% using accident IDs.
    # The same ID will always receive the same selection.
    ids = df["collision_index"].astype("string")
    hashes = pd.util.hash_pandas_object(ids, index=False)
    sample = df.loc[hashes % 20 == 0].copy()

    train.append(sample[sample["collision_year"] <= 2018])
    validation.append(sample[
        sample["collision_year"].between(2019, 2022)
    ])
    test.append(sample[
        sample["collision_year"].between(2023, 2025)
    ])

    print("Processed:", file.name, flush=True)

for name, parts in [
    ("train", train),
    ("validation", validation),
    ("test", test)
]:
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(OUTPUT / f"{name}.parquet", index=False)
    print(name, "rows:", len(result))

print("DATA PREPARATION COMPLETED")
