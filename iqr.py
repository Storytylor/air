import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# Before
plt.figure(figsize=(7,5))
df.boxplot(column=['SepalLength'])
plt.title("Before Outlier Removal")
plt.show()

# IQR
Q1 = df['SepalLength'].quantile(0.25)
Q3 = df['SepalLength'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df['SepalLength'] >= lower) & (df['SepalLength'] <= upper)]

# After
plt.figure(figsize=(7,5))
df_clean.boxplot(column=['SepalLength'])
plt.title("After Outlier Removal")
plt.show()

print(df_clean)
