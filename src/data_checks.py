
import pandas as pd
from pathlib import Path

# Load datasets
df = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 1. Check duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_count)

# 2. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Check leaking column
leaking_column = "spoilage_found_at_unloading"
print("\nLeaking column in train:", leaking_column in df.columns)
print("Leaking column in holdout:", leaking_column in holdout.columns)

# 4. Check spelling variants
print("\nOriginal crop values:")
print(df["crop"].value_counts().to_string())

crop_clean = (
    df["crop"]
    .str.strip()
    .str.lower()
    .replace({
        "app.": "apple",
        "pot.": "potato",
        "oni.": "onion"
    })
)

print("\nClean crop categories:")
print(crop_clean.value_counts())
print("Number of true crop categories:", crop_clean.nunique())


# 5. Check sentinel values
numeric_columns = df.select_dtypes(include="number").columns
sentinel_candidates = [-1, -999, 999, 9999]

print("\nSentinel value checks:")
for column in numeric_columns:
    for value in sentinel_candidates:
        count = (df[column] == value).sum()
        if count > 0:
            print(column, value, count)
# 6. Check possible unit mix-ups across chambers
unit_columns = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day"
]

print("\nMedian values by chamber:")
print(df.groupby("chamber")[unit_columns].median().to_string())

# 7. Check stuck sensor
df["date"] = pd.to_datetime(df["date"])

march_data = df[
    (df["date"] >= "2026-03-04") &
    (df["date"] <= "2026-03-27")
]

print("\nDoor opening readings during suspected stuck sensor period:")
print(march_data.groupby("date")["door_openings_per_day"].agg(
    ["min", "max", "nunique"]
).to_string())

# 8. Check target values for possible 10x typos
target_limit = df["spoilage_pct"].median() * 10
target_typos = df[df["spoilage_pct"] > target_limit]

print("\nPossible 10x target typos:")
print(target_typos[["record_id", "spoilage_pct"]].to_string(index=False))
print("Possible typo count:", len(target_typos))

print("\nData checks completed.")
