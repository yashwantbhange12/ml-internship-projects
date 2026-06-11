# ============================================================
# PROJECT: Fraud Detection System
# AUTHOR : [Your Name]
# DATE   : [Date]
# DATASET: Credit Card Fraud Detection (Kaggle)
#          https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================
# HOW TO RUN:
#   1. Download dataset from Kaggle link above
#   2. Place 'creditcard.csv' in same folder
#   3. pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
#   4. python fraud_detection.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Dataset")
print("=" * 55)

df = pd.read_csv('creditcard.csv')

print(f"Dataset Shape : {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumns: {list(df.columns)}")

# ─────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Exploratory Data Analysis")
print("=" * 55)

print("\nMissing Values:")
print(df.isnull().sum().sum(), "missing values found")

fraud_count = df['Class'].value_counts()
print(f"\nClass Distribution:")
print(fraud_count)
print(f"Fraud Rate: {fraud_count[1] / len(df) * 100:.4f}%")
# Note: This dataset is highly imbalanced — only ~0.17% are fraud!

# Plot 1: Class Distribution
plt.figure(figsize=(5, 4))
fraud_count.plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Transaction Class Distribution')
plt.xlabel('Class (0=Normal, 1=Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fraud_class_distribution.png')
plt.close()
print("\n[Saved] fraud_class_distribution.png")

# Plot 2: Transaction Amount by Class
plt.figure(figsize=(7, 4))
df.boxplot(column='Amount', by='Class', grid=False)
plt.suptitle('')
plt.title('Transaction Amount by Class')
plt.xlabel('Class (0=Normal, 1=Fraud)')
plt.ylabel('Amount')
plt.tight_layout()
plt.savefig('amount_by_class.png')
plt.close()
print("[Saved] amount_by_class.png")

# Plot 3: Fraud transactions over time
plt.figure(figsize=(8, 4))
fraud_df = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0]
plt.scatter(normal_df['Time'], normal_df['Amount'],
            alpha=0.3, s=2, label='Normal', color='steelblue')
plt.scatter(fraud_df['Time'], fraud_df['Amount'],
            alpha=0.6, s=5, label='Fraud', color='tomato')
plt.title('Transactions Over Time (Amount)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amount')
plt.legend()
plt.tight_layout()
plt.savefig('transactions_over_time.png')
plt.close()
print("[Saved] transactions_over_time.png")

# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Preprocessing")
print("=" * 55)

# Scale 'Amount' and 'Time' — V1-V28 are already PCA-scaled
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_Time']   = scaler.fit_transform(df[['Time']])

# Drop original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

print("Scaled 'Amount' and 'Time' columns.")
print(f"Final Shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 4: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Train-Test Split")
print("=" * 55)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]} rows")
print(f"Test size  : {X_test.shape[0]} rows")
print(f"Fraud in test set: {sum(y_test == 1)}")

# ─────────────────────────────────────────────
# STEP 5: HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Handling Imbalance with SMOTE")
print("=" * 55)

print(f"Before SMOTE - Normal: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE  - Normal: {sum(y_train_sm==0)}, Fraud: {sum(y_train_sm==1)}")

# ─────────────────────────────────────────────
# STEP 6: TRAIN CLASSIFICATION MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Training Classification Models")
print("=" * 55)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100,
                                                   random_state=42,
                                                   class_weight='balanced')
}

results = {}

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'accuracy': acc, 'f1': f1, 'auc': auc}

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}  |  F1-Score: {f1:.4f}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Normal', 'Fraud']))

# ─────────────────────────────────────────────
# STEP 7: ISOLATION FOREST (Anomaly Detection)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Isolation Forest (Anomaly Detection)")
print("=" * 55)

# Note: Isolation Forest is unsupervised — it does NOT use SMOTE data
iso = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
iso.fit(X_train)

iso_pred = iso.predict(X_test)
# Isolation Forest returns -1 for anomaly, 1 for normal — convert to 0/1
iso_pred = [1 if p == -1 else 0 for p in iso_pred]

iso_f1  = f1_score(y_test, iso_pred)
iso_auc = roc_auc_score(y_test, iso_pred)
print(f"Isolation Forest | F1: {iso_f1:.4f} | AUC: {iso_auc:.4f}")
print(classification_report(y_test, iso_pred,
                             target_names=['Normal', 'Fraud']))

# ─────────────────────────────────────────────
# STEP 8: CONFUSION MATRIX (Best Model)
# ─────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['auc'])
print(f"\nBest Classification Model by AUC: {best_name}")

cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_fraud.png')
plt.close()
print("[Saved] confusion_matrix_fraud.png")

# ─────────────────────────────────────────────
# STEP 9: ROC CURVE
# ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Fraud Detection')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_fraud.png')
plt.close()
print("[Saved] roc_curve_fraud.png")

# ─────────────────────────────────────────────
# STEP 10: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

plt.figure(figsize=(7, 5))
top10.sort_values().plot(kind='barh', color='tomato')
plt.title('Top 10 Important Features (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_fraud.png')
plt.close()
print("[Saved] feature_importance_fraud.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"{'Model':<25} | {'Accuracy':>8} | {'F1-Score':>8} | {'AUC':>6}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<25} | {res['accuracy']:>8.4f} | {res['f1']:>8.4f} | {res['auc']:>6.4f}")
print(f"{'Isolation Forest':<25} | {'N/A':>8} | {iso_f1:>8.4f} | {iso_auc:>6.4f}")

print(f"\nBest Model : {best_name}")
print("Output files saved: fraud_class_distribution.png, confusion_matrix_fraud.png,")
print("                    roc_curve_fraud.png, feature_importance_fraud.png")
print("\nProject Complete!")
