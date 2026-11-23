import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from feature_engineering import create_features

ACCOUNT = "amazonlake60301190"
SOURCE = "projects/road-accident-severity/processed_v1/"
DEST = "projects/road-accident-severity/features_v1/"

def azure(*args):
    result = subprocess.run(
        ["az", "storage", "blob", *args, "--auth-mode", "login"],
        check=True, capture_output=True, text=True
    )
    return result.stdout

# Find all processed Parquet files
files = json.loads(azure(
    "list", "--account-name", ACCOUNT,
    "--container-name", "processed",
    "--prefix", SOURCE, "--output", "json"
))
names = sorted(
    file["name"] for file in files
    if file["name"].endswith(".parquet")
)

assert len(names) == 183, f"Expected 183 files, found {len(names)}"

# Do not overwrite an existing feature version
existing = json.loads(azure(
    "list", "--account-name", ACCOUNT,
    "--container-name", "curated",
    "--prefix", DEST, "--output", "json"
))
if existing:
    raise RuntimeError("features_v1 already contains files. Upload stopped.")

total_input = 0
total_output = 0

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)

    for number, name in enumerate(names, 1):
        local_input = folder / "input.parquet"
        local_output = folder / "output.parquet"

        azure(
            "download", "--account-name", ACCOUNT,
            "--container-name", "processed",
            "--name", name, "--file", str(local_input),
            "--overwrite", "true", "--output", "none"
        )

        df = pd.read_parquet(local_input)
        result = create_features(df)

        assert len(df) == len(result)
        total_input += len(df)
        total_output += len(result)

        result.to_parquet(local_output, index=False)

        azure(
            "upload", "--account-name", ACCOUNT,
            "--container-name", "curated",
            "--name", DEST + Path(name).name,
            "--file", str(local_output),
            "--overwrite", "false", "--output", "none"
        )

        print(f"File {number}/183 completed", flush=True)

    # Save the summary in Azure with the feature files
    summary = {
        "source": SOURCE,
        "version": "features_v1",
        "files": len(names),
        "input_rows": total_input,
        "output_rows": total_output,
        "features": [
            "hour", "month", "is_weekend",
            "is_night", "high_speed_road"
        ]
    }

    summary_file = folder / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    azure(
        "upload", "--account-name", ACCOUNT,
        "--container-name", "curated",
        "--name", DEST + "summary.json",
        "--file", str(summary_file),
        "--overwrite", "false", "--output", "none"
    )

print("FEATURE ENGINEERING COMPLETED")
print(json.dumps(summary, indent=2))
