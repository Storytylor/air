import pandas as pd
import numpy as np

data = {
    'Age': [25, 30, np.nan, 35, 40],
    'Salary': [50000, np.nan, 60000, 65000, np.nan],
    'Marks': [80, 85, 90, np.nan, 95]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nAfter Dropping Rows:")
print(df.dropna())

print("\nAfter Dropping Columns:")
print(df.dropna(axis=1))

df_mean = df.fillna(df.mean(numeric_only=True))
print("\nAfter Mean Imputation:")
print(df_mean)

df_median = df.fillna(df.median(numeric_only=True))
print("\nAfter Median Imputation:")
print(df_median)
