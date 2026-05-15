import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# (i) Data loading
# -----------------------------
df = pd.read_csv("iris.csv")

print("Dataset:")
print(df.head())

# Features and target
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

# Encode target
y = y.astype('category').cat.codes

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# (ii) Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# (iii) Model training
# -----------------------------
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -----------------------------
# (iv) Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, average='macro'))

print("Recall:", recall_score(y_test, y_pred, average='macro'))

print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
