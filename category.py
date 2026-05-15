import pandas as pd

# Load dataset
df = pd.read_csv("/content/Iris.csv")

print("Original Data:")
print(df.head())

# --------------------
# 1. Label Encoding
# --------------------
df_label = df.copy()

df_label['Species'] = df_label['Species'].astype('category').cat.codes

print("\nAfter Label Encoding:")
print(df_label.head())

# --------------------
# 2. One Hot Encoding
# --------------------
df_onehot = pd.get_dummies(df, columns=['Species'])

print("\nAfter One Hot Encoding:")
print(df_onehot.head())
