# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv(r"D:\Data_Science\Credit_card\creditcard.csv")

# 2. Dataset info
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# 3. Class distribution
print("\nClass distribution (0 = genuine, 1 = fraud):")
print(df["Class"].value_counts())
fraud_percentage = (df["Class"].value_counts()[1] / len(df)) * 100
print(f"\nFraud transactions percentage: {fraud_percentage:.4f}%")

# 4. Train-test split
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# 5. Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE, training class distribution:")
print(y_train_res.value_counts())

# 6. Train Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
# 8. Evaluation
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 9. Confusion Matrix Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
