import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# Load dataset
df = pd.read_csv("/content/Iris.csv")

# Select numeric feature
col = 'SepalLengthCm'

# Calculate skewness
print("Skewness:")
print(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].skew())

# Original plot
plt.figure(figsize=(7,5))
plt.hist(df[col], bins=10)
plt.title("Before Transformation")
plt.xlabel(col)
plt.ylabel("Frequency")
plt.show()

# Log transformation
df['Log_Transform'] = np.log(df[col])

# Square root transformation
df['Sqrt_Transform'] = np.sqrt(df[col])

# Box-Cox transformation
df['BoxCox_Transform'], _ = boxcox(df[col])

# After transformation
plt.figure(figsize=(7,5))
plt.hist(df['BoxCox_Transform'], bins=10)
plt.title("After Box-Cox Transformation")
plt.xlabel("Transformed SepalLength")
plt.ylabel("Frequency")
plt.show()

print("\nSkewness after transformation:")
print("Log:", df['Log_Transform'].skew())
print("Sqrt:", df['Sqrt_Transform'].skew())
print("Box-Cox:", df['BoxCox_Transform'].skew())
