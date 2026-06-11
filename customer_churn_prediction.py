# ============================================================
# PROJECT: Customer Churn Prediction
# AUTHOR : [Your Name]
# DATE   : [Date]
# DATASET: Telco Customer Churn (Kaggle)
#          https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# ============================================================
# HOW TO RUN:
#   1. Download dataset from Kaggle link above
#   2. Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in same folder
#   3. pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
#   4. python customer_churn_prediction.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Dataset")
print("=" * 55)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Dataset Shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# STEP 2: BASIC EXPLORATION (EDA)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Exploratory Data Analysis")
print("=" * 55)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

# Plot 1: Churn Count
plt.figure(figsize=(5, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.close()
print("\n[Saved] churn_distribution.png")

# Plot 2: Monthly Charges vs Churn
plt.figure(figsize=(7, 4))
df.boxplot(column='MonthlyCharges', by='Churn', grid=False)
plt.suptitle('')
plt.title('Monthly Charges by Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.tight_layout()
plt.savefig('monthly_charges_churn.png')
plt.close()
print("[Saved] monthly_charges_churn.png")

# Plot 3: Tenure vs Churn
plt.figure(figsize=(7, 4))
df[df['Churn'] == 'Yes']['tenure'].hist(alpha=0.6, label='Churned', bins=30, color='tomato')
df[df['Churn'] == 'No']['tenure'].hist(alpha=0.6, label='Stayed', bins=30, color='steelblue')
plt.title('Tenure Distribution by Churn')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('tenure_churn.png')
plt.close()
print("[Saved] tenure_churn.png")

# ─────────────────────────────────────────────
# STEP 3: DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Data Preprocessing")
print("=" * 55)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Fix TotalCharges - it has spaces, convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target variable: Yes=1, No=0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode all other categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
print(f"\nEncoding {len(cat_cols)} categorical columns: {list(cat_cols)}")

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Encoding done.")
print(f"\nFinal Shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 4: SPLIT DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Train-Test Split")
print("=" * 55)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]} rows")
print(f"Test size  : {X_test.shape[0]} rows")

# ─────────────────────────────────────────────
# STEP 5: HANDLE CLASS IMBALANCE USING SMOTE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Handling Imbalance with SMOTE")
print("=" * 55)

print(f"Before SMOTE - Churn=0: {sum(y_train==0)}, Churn=1: {sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE  - Churn=0: {sum(y_train_sm==0)}, Churn=1: {sum(y_train_sm==1)}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 6: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Training Models")
print("=" * 55)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train_sm)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'accuracy': acc, 'auc': auc}

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# STEP 7: CONFUSION MATRIX (Best Model)
# ─────────────────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]['auc'])
print(f"\nBest Model by AUC: {best_model_name}")

cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_churn.png')
plt.close()
print("[Saved] confusion_matrix_churn.png")

# ─────────────────────────────────────────────
# STEP 8: ROC CURVE (All Models)
# ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_churn.png')
plt.close()
print("[Saved] roc_curve_churn.png")

# ─────────────────────────────────────────────
# STEP 9: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

plt.figure(figsize=(7, 5))
top10.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 10 Important Features (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_churn.png')
plt.close()
print("[Saved] feature_importance_churn.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
for name, res in results.items():
    print(f"{name:25s} | Accuracy: {res['accuracy']:.4f} | AUC: {res['auc']:.4f}")

print(f"\nBest Model : {best_model_name}")
print("Output files saved: churn_distribution.png, confusion_matrix_churn.png,")
print("                    roc_curve_churn.png, feature_importance_churn.png")
print("\nProject Complete!")
