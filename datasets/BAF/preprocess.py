import numpy as np
import pandas as pd

read_path = "Base.csv"
write_file_name = "Base_processed"
cat_cols = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
]

df = pd.read_csv(read_path)

# Encode categorical columns
for col in cat_cols:
    vc = df[col].value_counts()
    d = {cat: idx for idx, cat in enumerate(vc.index)}
    df[col] = df[col].map(d)

# Create timestamp from `month` column
timestamps = [  # 2022
    1643673600,  # feb
    1646092800,  # mar
    1648771200,  # apr
    1651363200,  # may
    1654041600,  # jun
    1656633600,  # jul
    1659312000,  # aug
    1661990400,  # sep
    1664582400,  # oct
]
df["timestamp"] = 0
for month in range(8):
    idx = df["month"] == month
    count = idx.sum()
    df.loc[idx, "timestamp"] = np.linspace(
        timestamps[month], timestamps[month + 1], count, endpoint=False
    ).astype(int)

# Write Base_enc.csv
df.to_csv(write_file_name + ".csv", index=False)

# Create and write splits using `device_os` column
for value in df["device_os"].unique():
    dfs = df.loc[df["device_os"] == value]
    dfs.to_csv(f"{write_file_name}_{value}.csv", index=False)
