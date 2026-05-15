import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# (i) Reading data
# -------------------------------
df = pd.read_csv("/content/Iris.csv")

print("Original Dataset:")
print(df.head())

# -------------------------------
# (ii) Handling missing values
# -------------------------------
# (Example for demonstration)
df.loc[2, 'SepalLengthCm'] = np.nan
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------------
# (iii) Handling outliers (IQR)
# -------------------------------
col = 'SepalLengthCm'

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df[col] >= lower) & (df[col] <= upper)]

# -------------------------------
# (iv) Skewness removal
# -------------------------------
df['SepalLengthCm'], _ = boxcox(df['SepalLengthCm'])

# -------------------------------
# (v) Handling categorical data
# -------------------------------
df['Species'] = df['Species'].astype('category').cat.codes

# -------------------------------
# (vi) Feature engineering
# -------------------------------
# create new feature
df['SepalArea'] = df['SepalLengthCm'] * df['SepalWidthCm']

# -------------------------------
# (vii) StandardScaler
# -------------------------------
X = df.drop('Species', axis=1)
y = df['Species']

# -------------------------------
# (viii) Splitting data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# (ix) Importing and training model
# -------------------------------
model = RandomForestClassifier()

model.fit(X_train, y_train)

# -------------------------------
# (x) Model evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
