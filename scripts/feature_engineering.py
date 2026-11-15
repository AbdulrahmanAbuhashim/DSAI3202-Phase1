import pandas as pd

def create_features(df):
    df = df.copy()

    # Convert date and time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(
        df["time"].astype("string").str.split(":").str[0],
        errors="coerce"
    )

    # Time-related features
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)

    # Road-related features
    df["speed_limit"] = pd.to_numeric(df["speed_limit"], errors="coerce")
    df["high_speed_road"] = (df["speed_limit"] >= 50).astype(int)

    # Preserve categorical road and weather features
    for col in ["weather_conditions", "light_conditions",
                "road_surface_conditions", "road_type"]:
        df[col] = df[col].astype("string")

    return df
