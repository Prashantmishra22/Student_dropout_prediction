
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Objective 2: Predict Student Outcomes using KNN
# To apply the K-Nearest Neighbors algorithm to the student dataset and classify academic outcomes based on similarities between student records. The model will use standardized features and the selected number of neighbors, and its predictive performance will be evaluated using appropriate classification metrics.

# Load Dataset
df = pd.read_csv("data.csv", sep=";")

# Data Preprocessing
print("First five rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove Duplicate Rows
df = df.drop_duplicates()

# Handle Missing Values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Feature Engineering
df["Total_Approved_Units"] = (
    df["Curricular units 1st sem (approved)"]
    + df["Curricular units 2nd sem (approved)"]
)

print("\nFeature Engineering:")
print(df["Total_Approved_Units"].head())

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create KNN Model
model = KNeighborsClassifier(
    n_neighbors=7,
    metric="euclidean"
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred[:10])

# Accuracy
print("\nKNN Accuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
