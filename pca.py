import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.csv")

# Features
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# New dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1','PC2'])

print("Reduced Dataset:")
print(pca_df.head())

# Plot
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.show()

# Explained variance
print("\nVariance Ratio:")
print(pca.explained_variance_ratio_)
