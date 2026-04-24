"""
SCALA-Guard ML Model Training v2.5 (Improved)
Advanced features + Ensemble methods + Real-world simulation
Run: python train_model_v2_improved.py
"""
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("  SCALA-Guard ML Model Training v2.5 (IMPROVED)")
print("  Features: 12 advanced indicators | Models: 3 ensemble methods")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: ENHANCED SYNTHETIC DATASET WITH 12 FEATURES
# ═══════════════════════════════════════════════════════════════════════════

np.random.seed(42)
N_SAMPLES = 5000  # Increased from 2000 to 5000

print(f"\n📊 Generating {N_SAMPLES} synthetic samples with 12 features...")

# ─── BENIGN PACKAGES (label=0) ─────────────────────────────────────────────
benign_data = {
    "syscall_count": np.random.randint(50, 300, N_SAMPLES // 2),
    "network_connections": np.random.randint(0, 3, N_SAMPLES // 2),
    "file_access_count": np.random.randint(0, 5, N_SAMPLES // 2),
    "data_exfiltrated_kb": np.random.uniform(0, 10, N_SAMPLES // 2),
    "process_spawn_count": np.random.randint(0, 2, N_SAMPLES // 2),
    
    # NEW: Advanced features
    "entropy_score": np.random.uniform(3.0, 5.5, N_SAMPLES // 2),  # File entropy (low = normal)
    "dns_queries": np.random.randint(0, 5, N_SAMPLES // 2),        # DNS lookups
    "privilege_escalation_attempts": np.zeros(N_SAMPLES // 2, dtype=int),
    "registry_modifications": np.random.randint(0, 3, N_SAMPLES // 2),  # Windows registry
    "dll_injection_count": np.zeros(N_SAMPLES // 2, dtype=int),
    "suspicious_imports": np.random.randint(0, 2, N_SAMPLES // 2),
    "command_execution": np.zeros(N_SAMPLES // 2, dtype=int),
}

# ─── MALICIOUS PACKAGES (label=1) ──────────────────────────────────────────
malicious_data = {
    "syscall_count": np.random.randint(400, 2500, N_SAMPLES // 2),
    "network_connections": np.random.randint(3, 15, N_SAMPLES // 2),
    "file_access_count": np.random.randint(5, 20, N_SAMPLES // 2),
    "data_exfiltrated_kb": np.random.uniform(50, 1000, N_SAMPLES // 2),
    "process_spawn_count": np.random.randint(1, 8, N_SAMPLES // 2),
    
    # NEW: Advanced features with malicious patterns
    "entropy_score": np.random.uniform(6.5, 7.9, N_SAMPLES // 2),
    "dns_queries": np.random.randint(10, 50, N_SAMPLES // 2),
    "privilege_escalation_attempts": np.random.randint(1, 5, N_SAMPLES // 2),
    "registry_modifications": np.random.randint(10, 50, N_SAMPLES // 2),
    "dll_injection_count": np.random.randint(1, 10, N_SAMPLES // 2),
    "suspicious_imports": np.random.randint(5, 15, N_SAMPLES // 2),
    "command_execution": np.random.randint(1, 8, N_SAMPLES // 2),
}

# Stack features
feature_names = list(benign_data.keys())
X_benign = np.column_stack([benign_data[f] for f in feature_names])
X_malicious = np.column_stack([malicious_data[f] for f in feature_names])

X = np.vstack([X_benign, X_malicious])
y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))

print(f"   Features: {feature_names}")
print(f"   Benign samples: {len(X_benign)}")
print(f"   Malicious samples: {len(X_malicious)}")

# ─── Train/Test/Validation Split ──────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

print(f"\n📈 Dataset Split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: MODEL TRAINING - THREE ENSEMBLE METHODS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔧 Training ensemble models...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ─── Model 1: Random Forest (Tuned) ────────────────────────────────────────
print("\n   1️⃣  Random Forest (Hyperparameter tuning)...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [3, 5, 7],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    rf_params, cv=5, scoring='f1'
)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
print(f"      Best params: {rf_grid.best_params_}")
print(f"      CV F1 Score: {rf_grid.best_score_:.4f}")

# ─── Model 2: Gradient Boosting (XGBoost-style) ────────────────────────────
print("\n   2️⃣  Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
gb_val_score = gb.score(X_val_scaled, y_val)
print(f"      Validation Accuracy: {gb_val_score:.4f}")

# ─── Model 3: AdaBoost (Adaptive Boosting) ────────────────────────────────
print("\n   3️⃣  AdaBoost Classifier...")
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.8,
    random_state=42
)
ada.fit(X_train_scaled, y_train)
ada_val_score = ada.score(X_val_scaled, y_val)
print(f"      Validation Accuracy: {ada_val_score:.4f}")

# ─── Ensemble: Voting Classifier ───────────────��──────────────────────────
print("\n   🎯 Creating Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('gb', gb),
        ('ada', ada)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
ensemble_score = ensemble.score(X_val_scaled, y_val)
print(f"      Ensemble Validation Accuracy: {ensemble_score:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: EVALUATION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  📊 TEST SET EVALUATION")
print("=" * 70)

models = {
    "Random Forest": rf_best,
    "Gradient Boosting": gb,
    "AdaBoost": ada,
    "Ensemble (Voting)": ensemble
}

best_f1 = 0
best_model_name = ""
results_summary = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    results_summary[name] = {
        "f1": f1,
        "auc": auc_score,
        "predictions": y_pred
    }
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model
        best_y_pred = y_pred
        best_y_proba = y_proba
    
    print(f"\n{name}:")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc_score:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'])}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  🎯 FEATURE IMPORTANCE (Top 5)")
print("=" * 70)

# Random Forest feature importance
rf_importance = rf_best.feature_importances_
importance_df = sorted(
    zip(feature_names, rf_importance),
    key=lambda x: x[1],
    reverse=True
)

print("\nRandom Forest Importance Ranking:")
for i, (feature, importance) in enumerate(importance_df[:5], 1):
    bar = "█" * int(importance * 50)
    print(f"  {i}. {feature:30s} {bar} {importance:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 5: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  💾 SAVING MODELS")
print("=" * 70)

# Save best model
joblib.dump(best_model, "scala_guard_model_ensemble.pkl")
print(f"\n✅ Best Model Saved: scala_guard_model_ensemble.pkl")
print(f"   Model: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")

# Save individual models
joblib.dump(rf_best, "scala_guard_model_rf.pkl")
joblib.dump(gb, "scala_guard_model_gb.pkl")
joblib.dump(ada, "scala_guard_model_ada.pkl")
print(f"✅ Individual models saved (RF, GB, Ada)")

# Save scaler
joblib.dump(scaler, "scala_guard_scaler.pkl")
print(f"✅ Feature scaler saved: scala_guard_scaler.pkl")

# Save feature names
with open("scala_guard_features.txt", "w") as f:
    f.write("\n".join(feature_names))
print(f"✅ Feature names saved: scala_guard_features.txt")

# ═══════════════════════════════════════════════════════════════════════════
# PART 6: CONFUSION MATRIX & METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  📊 CONFUSION MATRIX & DETAILED METRICS")
print("=" * 70)

cm = confusion_matrix(y_test, best_y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n  Sensitivity (Recall):     {sensitivity:.4f}")
print(f"  Specificity:              {specificity:.4f}")
print(f"  Precision:                {precision:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ✨ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Performance Metrics:")
print(f"   • F1-Score: {best_f1:.4f}")
print(f"   • ROC-AUC: {results_summary[best_model_name]['auc']:.4f}")
print(f"   • Top Feature: {importance_df[0][0]}")
print(f"\n📁 Saved files:")
print(f"   • scala_guard_model_ensemble.pkl (Best)")
print(f"   • scala_guard_model_rf.pkl")
print(f"   • scala_guard_model_gb.pkl")
print(f"   • scala_guard_model_ada.pkl")
print(f"   • scala_guard_scaler.pkl")
print(f"   • scala_guard_features.txt")
