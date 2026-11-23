import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from etl_transform import clean_batch

data = pd.DataFrame({
    "collision_index": ["A001", "", "A003", "A004"],
    "collision_year": [2024] * 4,
    "collision_severity": [1, 2, 9, 3],
    "date": ["01/01/2024", "01/01/2024", "01/01/2024", "invalid"],
    "speed_limit": [30] * 4,
    "longitude": [-0.1] * 4,
    "latitude": [51.5] * 4
})

accepted, rejected = clean_batch(data)

assert len(accepted) == 1
assert len(rejected) == 3

expected = {
    "missing_collision_index",
    "invalid_collision_severity",
    "invalid_date"
}
assert set(rejected["rejection_reason"]) == expected

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "rejected.parquet"
    rejected.to_parquet(path, index=False)
    saved = pd.read_parquet(path)

    assert len(saved) == 3
    assert set(saved["rejection_reason"]) == expected

print("Accepted:", len(accepted))
print("Rejected:", len(rejected))
print("Rejection reasons:", sorted(expected))
print("ETL REJECTION TEST PASSED")
