import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("/content/Iris.csv")

# Convert categorical target to numeric
df['Species'] = df['Species'].astype('category').cat.codes

print("Dataset:")
print(df.head())

# -------------------------
# 1. Heatmap
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 2. Top 5 correlated features
# -------------------------
corr = df.corr()['Species'].abs().sort_values(ascending=False)

print("\nTop 5 correlated features:")
print(corr.head(5))

# -------------------------
# 3. Train Test Split
# -------------------------
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)
