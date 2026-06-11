# Machine Learning Projects — Codec Technologies Internship

Two beginner-level ML projects completed as part of the internship program.

---

## Project 1: Customer Churn Prediction

**Goal:** Predict which customers are likely to leave a telecom service.

### Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Techniques Used
- Exploratory Data Analysis (EDA)
- Label Encoding for categorical variables
- SMOTE to handle class imbalance
- Models: Logistic Regression, Decision Tree, Random Forest
- Evaluation: Accuracy, F1-Score, AUC-ROC, Confusion Matrix

### How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python customer_churn_prediction.py
```

### Results
| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | ~0.79    | ~0.84   |
| Decision Tree       | ~0.78    | ~0.79   |
| Random Forest       | ~0.82    | ~0.87   |

---

## Project 2: Fraud Detection System

**Goal:** Detect fraudulent credit card transactions from highly imbalanced data.

### Dataset
[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Techniques Used
- Feature scaling (StandardScaler on Amount and Time)
- SMOTE for class imbalance (~0.17% fraud rate)
- Models: Logistic Regression, Random Forest, Isolation Forest (anomaly detection)
- Evaluation: F1-Score, AUC-ROC, Confusion Matrix

> **Note:** Accuracy is misleading here due to class imbalance. F1-Score and AUC-ROC are better metrics.

### How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python fraud_detection.py
```

### Results
| Model               | F1-Score | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | ~0.72    | ~0.97   |
| Random Forest       | ~0.86    | ~0.99   |
| Isolation Forest    | ~0.25    | ~0.61   |

---

## Folder Structure
```
├── customer_churn_prediction.py
├── fraud_detection.py
├── README.md
└── outputs/
    ├── churn_distribution.png
    ├── confusion_matrix_churn.png
    ├── roc_curve_churn.png
    ├── fraud_class_distribution.png
    ├── confusion_matrix_fraud.png
    └── roc_curve_fraud.png
```

## Tools & Libraries
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn

---

*Internship Project — Codec Technologies*
