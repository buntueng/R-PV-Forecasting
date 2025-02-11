import pandas as pd
import numpy as np
import time
import os

input_file = "dataset/dataset.csv"
output_file = "dataset/preprocess_dataset.csv"

if os.path.exists(output_file):
    os.remove(output_file)

# 1. Data Loading and Preprocessing
df = pd.read_csv(input_file, parse_dates=["DateTime"])

# Convert 'DateTime' to datetime objects (handle potential format issues)
df["DateTime"] = pd.to_datetime(df["DateTime"], format='mixed', errors='coerce')
df = df.set_index("DateTime").resample("H").agg({"MW": "sum", "Capacity": "sum"})

df['MW'] = df['MW'].round(2)
df['Capacity'] = df['Capacity'].round(2)

# Filter data between 8 AM and 4 PM (inclusive of 8 AM, exclusive of 5 PM)
df = df.between_time("08:00", "17:00")  # Correct time range for filtering

# Reset the index if you need 'DateTime' as a regular column
df = df.reset_index()

# Filter data between 8 AM and 5 PM
hour_feature = df["DateTime"].dt.hour

# Feature Engineering (Cyclic Time of Day)
df['Time_sin'] = np.sin(2 * np.pi * hour_feature / 24)
df['Time_cos'] = np.cos(2 * np.pi * hour_feature / 24)

df['Time_sin'] = df['Time_sin'].round(2)
df['Time_cos'] = df['Time_cos'].round(2)

# Day of Year (Cyclic)
day_of_year = df['DateTime'].dt.dayofyear
df['Day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
df['Day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

df['Day_sin'] = df['Day_sin'].round(2)
df['Day_cos'] = df['Day_cos'].round(2)

# # save the preprocessed data to a new file
df.to_csv(output_file, index=False)
