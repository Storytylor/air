import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Convert target numbers to species names
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Set style
sns.set(style="whitegrid")

# 1. Histogram
plt.figure(figsize=(8,5))
plt.hist(df['sepal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.legend(["Sepal Length"])
plt.show()

# 2. Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title("Boxplot of Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# 4. Bar Chart
plt.figure(figsize=(8,5))
df['species'].value_counts().plot(kind='bar')
plt.title("Count of Each Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.legend(["Species Count"])
plt.show()
