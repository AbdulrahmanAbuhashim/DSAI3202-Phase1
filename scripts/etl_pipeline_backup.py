import json
import csv
from pathlib import Path

PROJECT_DIR = Path.home() / "DSAI3202-Phase1"
CONFIG_PATH = PROJECT_DIR / "config" / "etl_config.json"

def main():
    with CONFIG_PATH.open("r") as f:
        config = json.load(f)

    source = Path(config["input_csv"])

    print("=== DSAI3202 PHASE 1 ETL ===")
    print("Project:", config["project"])
    print("Source version:", config["source_version"])
    print("Output version:", config["output_version"])

    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")

    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        columns = next(reader)

    missing = [
        col for col in config["required_columns"]
        if col not in columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Input CSV:", source.name)
    print("Source size:", source.stat().st_size, "bytes")
    print("Total columns:", len(columns))
    print("Required columns:", len(config["required_columns"]))
    print("Schema validation: PASSED")
    print("Configuration validation: PASSED")

if __name__ == "__main__":
    main()
