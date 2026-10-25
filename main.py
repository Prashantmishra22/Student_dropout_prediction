
# “My project starts with a university that wants to understand student academic outcomes. I use student data, apply four machine learning classifiers, compare their results, and identify the best-performing model. The goal is to understand student outcomes and explore how data-driven insights could support students.”


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score

# Student Dropout Prediction - Predictive Analysis Project

# To implement a Gaussian Naive Bayes classifier that estimates the probabilities of the three student outcome categories using the available features. The model will be evaluated to understand its classification performance and compare its results with the other algorithms.

# 1. Load Dataset
df = pd.read_csv("data.csv", sep=";")

print("First Five Rows:")
print(df.head())

print("\nLast Five Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 2. Data Cleaning

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Feature Engineering
df["Total_Approved_Units"] = (
    df["Curricular units 1st sem (approved)"]
    + df["Curricular units 2nd sem (approved)"]
)

print("\nNew Feature:")
print(df["Total_Approved_Units"].head())

# 4. Exploratory Data Analysis (EDA)

print("\nStudent Outcome Counts:")
print(df["Target"].value_counts())

# Graph 1: Target Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Target")
plt.title("Student Outcome Distribution")
plt.xlabel("Student Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Graph 2: Target Pie Chart
df["Target"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", figsize=(7, 7)
)
plt.title("Percentage of Student Outcomes")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Graph 3: Approved Units by Outcome
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Target",
    y="Total_Approved_Units"
)
plt.title("Approved Units by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Total Approved Units")
plt.tight_layout()
plt.show()

# Graph 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Prepare Data for Machine Learning

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Data for Logistic Regression and KNN
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# 6. Create Four Machine Learning Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        metric="euclidean"
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

results = []

# 7. Train and Evaluate Models

for name, model in models.items():

    print("\n===================================")
    print(name)
    print("===================================")

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1 Score": macro_f1
    })

    print("Accuracy:", accuracy)
    print("Macro F1 Score:", macro_f1)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Dropout", "Enrolled", "Graduate"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dropout", "Enrolled", "Graduate"],
        yticklabels=["Dropout", "Enrolled", "Graduate"]
    )
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 8. Compare All Models

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(
    results_df.sort_values(
        by="Macro F1 Score",
        ascending=False
    ).to_string(index=False)
)

# 9. Model Comparison Graph

plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)
plt.title("Accuracy Comparison of Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 10. Identify Best Model by Macro F1 Score

best_model = results_df.loc[
    results_df["Macro F1 Score"].idxmax()
]

print("\nBest Model Based on Macro F1 Score:")
print(best_model.to_string())

print("\nProject Analysis Completed!")

