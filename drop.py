import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")

# Add sample missing values
df.loc[2, 'SepalLength'] = np.nan
df.loc[5, 'PetalLength'] = np.nan

print("Original Data:")
print(df.head(10))

# Drop rows
print("\nDrop rows:")
print(df.dropna())

# Mean imputation
df_mean = df.copy()
df_mean['SepalLength'] = df_mean['SepalLength'].fillna(df_mean['SepalLength'].mean())
df_mean['PetalLength'] = df_mean['PetalLength'].fillna(df_mean['PetalLength'].mean())

print("\nMean Imputation:")
print(df_mean.head(10))
