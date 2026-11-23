import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from etl_transform import clean_batch
from etl_quality import check_quality
from etl_duplicates import DuplicateTracker

PROJECT = Path.home() / "DSAI3202-Phase1"
CONFIG = PROJECT / "config" / "etl_config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    with CONFIG.open() as f:
        config = json.load(f)

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)

    # Keep runs within the project output directory.
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("Invalid run name")

    base = PROJECT / "outputs"
    published = base / args.run_name
    staging = base / (args.run_name + "_staging")

    if published.exists() or staging.exists():
        raise FileExistsError(
            "Output already exists. Use a new run name."
        )

    staging.mkdir(parents=True)

    total = 0
    accepted_total = 0
    rejected_total = 0

    duplicate_total = 0
    tracker = DuplicateTracker(staging / "seen_ids.db")

    try:
        reader = pd.read_csv(
            source,
            dtype=str,
            chunksize=config["batch_size"]
        )

        for number, batch in enumerate(reader):
            if number == 0:
                missing = set(config["required_columns"]) - set(batch.columns)
                if missing:
                    raise ValueError(f"Missing columns: {sorted(missing)}")

            accepted, rejected = clean_batch(batch)

            # Detect duplicate IDs across all batches.
            flags = tracker.check_batch(
                accepted["collision_index"].tolist()
            )
            duplicates = accepted.loc[flags]
            accepted = accepted.loc[
                [not flag for flag in flags]
            ].copy()

                # Save rejected records with their reasons
            rejected_dir = staging / "rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)

            if len(rejected):
                rejected.to_parquet(
                    rejected_dir / f"invalid_{number:05d}.parquet",
                    index=False
                )

            if len(duplicates):
                duplicate_rows = duplicates.copy()
                duplicate_rows["rejection_reason"] = "duplicate_collision_index"
                duplicate_rows.to_parquet(
                    rejected_dir / f"duplicates_{number:05d}.parquet",
                    index=False
                )

            total += len(batch)
            accepted_total += len(accepted)
            rejected_total += len(rejected) + len(duplicates)
            duplicate_total += len(duplicates)

            if len(accepted):
                accepted.to_parquet(
                    staging / f"part_{number:05d}.parquet",
                    index=False
                )

            print(
                f"Batch {number + 1}: "
                f"{total:,} records processed, "
                f"{accepted_total:,} accepted, "
                f"{rejected_total:,} rejected",
                flush=True
            )

        tracker.close()
        (staging / "seen_ids.db").unlink()

        quality = check_quality(
            total,
            accepted_total,
            rejected_total,
            config["maximum_rejected_fraction"]
        )

        print("=== ETL RUN RESULTS ===")
        print("Input rows:", total)
        print("Accepted rows:", accepted_total)
        print("Rejected rows:", rejected_total)
        print("Duplicate rows:", duplicate_total)
        print(f"Rejection rate: {quality['rejection_rate']:.2%}")

        if not quality["publication_allowed"]:
            print("Quality gate: FAILED")
            print("Publication: BLOCKED")
            raise RuntimeError("Quality threshold exceeded")

        with (staging / "quality.json").open("w") as f:
            json.dump(quality, f, indent=2)

        staging.rename(published)

        print("Quality gate: PASSED")
        print("Publication: SUCCESS")
        print("Output:", published)

    except Exception:
        # Keep the failed run files for checking errors.
        print("ETL failed. Check the log and staging folder.")
        print("Staging folder:", staging)
        raise


if __name__ == "__main__":
    main()
