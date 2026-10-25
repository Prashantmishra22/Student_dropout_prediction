# 🎓 Student Dropout Prediction — Predictive Analysis

A machine learning project that predicts student academic outcomes (**Dropout**, **Enrolled**, or **Graduate**) using four classification algorithms. The goal is to understand student outcomes and explore how data-driven insights could support students at risk of dropping out.

---

## 📌 Problem Statement

A university wants to understand student academic outcomes. This project uses student data, applies four machine learning classifiers, compares their results, and identifies the best-performing model to support data-driven decision-making in higher education.

---

## 📊 Dataset

- **File:** `data.csv` (semicolon-separated)
- **Records:** ~4,424 students
- **Target Variable:** `Target` — 3 classes: `Dropout`, `Enrolled`, `Graduate`
- **Features:** 36 academic, demographic, and socioeconomic attributes including:
  - Curricular units (enrolled, approved, grades) for 1st & 2nd semester
  - Admission grade, previous qualification grade
  - Age at enrollment, gender, scholarship holder, debtor status
  - Tuition fees up to date, marital status, nationality
  - Unemployment rate, inflation rate, GDP

---

## 🤖 Machine Learning Models

Four classifiers were trained and evaluated:

| # | Model | Description |
|---|-------|-------------|
| 1 | **Logistic Regression** | Linear classifier with StandardScaler preprocessing (`max_iter=1000`) |
| 2 | **K-Nearest Neighbors (KNN)** | Distance-based classifier (`k=7`, Euclidean metric, StandardScaler) |
| 3 | **Decision Tree** | Rule-based classifier (`criterion=entropy`, no scaling needed) |
| 4 | **Gaussian Naive Bayes** | Probabilistic classifier (no scaling needed) |

---

## 📈 Results

| Model | Accuracy | Macro F1 Score |
|-------|----------|----------------|
| **Logistic Regression** | **76.95%** | **0.6849** |
| Decision Tree | 69.49% | 0.6338 |
| KNN | 68.02% | 0.5814 |
| Naive Bayes | 67.34% | 0.5801 |

> ✅ **Best Model:** Logistic Regression — highest accuracy and Macro F1 Score.

---

## 🗂️ Project Structure

```
Student_Dropout_Prediction_predective_analysis/
│
├── main.py                        # Full pipeline: EDA + all 4 models + comparison
├── firstclassifierregression.py   # Standalone Logistic Regression implementation
├── secondknn.py                   # Standalone KNN implementation
├── decision_tree.py               # Standalone Decision Tree implementation
├── naive_bayes.py                 # Standalone Naive Bayes implementation
├── data.csv                       # Student dataset
├── model_comparison.csv           # Model comparison results
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🔧 Pipeline Overview

1. **Data Loading** — Read semicolon-separated CSV
2. **Data Cleaning** — Remove duplicates, fill missing values (median for numeric, mode for categorical)
3. **Feature Engineering** — Create `Total_Approved_Units` (sum of 1st + 2nd semester approved units)
4. **Exploratory Data Analysis** — Target distribution, pie chart, box plots, correlation heatmap
5. **Preprocessing** — Train-test split (80/20, stratified), StandardScaler for LR & KNN
6. **Model Training & Evaluation** — Accuracy, Macro F1, classification report, confusion matrix
7. **Model Comparison** — Bar chart and best model identification by Macro F1 Score

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.8+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Prashantmishra22/Student_dropout_prediction.git
cd Student_dropout_prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Usage

```bash
# Run the full analysis pipeline (all 4 models + comparison)
python main.py

# Or run individual models
python firstclassifierregression.py   # Logistic Regression
python secondknn.py                   # KNN
python decision_tree.py               # Decision Tree
python naive_bayes.py                 # Naive Bayes
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |
| `scikit-learn` | Machine learning models and metrics |

---

## 📜 License

This project is open source and available for educational purposes.

---

## 👤 Author

**Prashant Mishra**

- GitHub: [@Prashantmishra22](https://github.com/Prashantmishra22)
