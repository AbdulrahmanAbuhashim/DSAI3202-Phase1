import pandas as pd


def clean_batch(df):
    """Clean and validate one batch of road accident records."""

    df = df.copy()

    # Convert numerical fields
    numeric_columns = [
        "collision_year",
        "collision_severity",
        "speed_limit",
        "longitude",
        "latitude"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse accident dates
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Preserve missing historical coordinates
    # rather than dropping valid accidents.

    # Define valid records
    valid_mask = (
        df["collision_index"].notna()
        & df["collision_index"].astype("string").str.strip().ne("").fillna(False)
        & df["collision_severity"].isin([1, 2, 3])
        & df["date"].notna()
    )

    accepted = df.loc[valid_mask].copy()
    rejected = df.loc[~valid_mask].copy()

    return accepted, rejected
