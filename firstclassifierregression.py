
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Objective 1: Predict Student Outcomes using Logistic Regression
# Objective 1: Logistic Regression

# To develop a Logistic Regression classification model using student-related data to predict whether a student belongs to the Graduate, Dropout, or Enrolled category. The model's performance will be evaluated using accuracy, precision, recall, F1-score, and a confusion matrix to understand how effectively it classifies student outcomes.

# Load Dataset
df = pd.read_csv("data.csv", sep=";")

# Data Preprocessing
print("First five rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

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

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred[:10])

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
