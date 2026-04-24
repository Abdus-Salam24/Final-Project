# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. Load Dataset
# =========================

DATA_PATH = "final_selected_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print(df.head())


# =========================
# 2. Features and Target
# =========================

TARGET_COLUMN = "label"

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print("\nSelected Features:")
print(X.columns.tolist())


# =========================
# 3. Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.6,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# =========================
# 4. Model Training
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")


# =========================
# 5. Model Evaluation
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 6. Save Model
# =========================

MODEL_PATH = "malicious_detection_model.pkl"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully -> {MODEL_PATH}")


# =========================
# 7. Save Feature Names
# =========================

FEATURES_PATH = "feature_columns.pkl"

joblib.dump(X.columns.tolist(), FEATURES_PATH)

print(f"Feature columns saved successfully -> {FEATURES_PATH}")