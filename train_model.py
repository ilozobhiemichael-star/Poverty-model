import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib

np.random.seed(42)

# ── Generate 10,000 rows ──────────────────────────────────────────
n = 10000

data = pd.DataFrame({
    "income":           np.random.randint(20000, 500000, n),
    "household_size":   np.random.randint(1, 10, n),
    "education_years":  np.random.randint(0, 20, n),
    "employment":       np.random.randint(0, 2, n),
    "financial_assets": np.random.randint(0, 1000000, n)
})

# ── Derived features ──────────────────────────────────────────────
data["income_per_person"] = data["income"] / data["household_size"]
data["assets_ratio"]      = data["financial_assets"] / (data["income"] + 1)

# ── Poverty label (threshold: ₦90,000 per person) ─────────────────
data["poverty"] = (
    (data["income_per_person"] < 90000) |
    (data["employment"] == 0) |
    (data["education_years"] < 6) |
    (data["assets_ratio"] < 0.5)
).astype(int)

# ── Features (order matters — must match app.py) ──────────────────
FEATURE_COLS = [
    "income", "household_size", "income_per_person",
    "education_years", "employment", "financial_assets"
]

X = data[FEATURE_COLS]
y = data["poverty"]

# ── Train / test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────
# ── Train (with hyperparameter tuning) ────────────────────────────
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 1, 2]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid.fit(X_train_scaled, y_train)

print("Best C:", grid.best_params_["C"])
print("Best CV Score:", grid.best_score_)

model = grid.best_estimator_

# ── Evaluation ────────────────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")

print("=" * 58)
print("   Poverty Risk Model — Evaluation Report")
print("=" * 58)
print(f"\n   Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"   Test Accuracy     : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"   ROC-AUC Score     : {roc_auc:.4f}")
print(f"   CV Score (5-fold) : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

overfit_gap = train_acc - test_acc
if overfit_gap > 0.05:
    print(f"\n   WARNING: Possible overfitting detected (gap = {overfit_gap:.4f})")
else:
    print(f"\n   Model generalises well (train/test gap = {overfit_gap:.4f})")

print("\n   Classification Report:\n")
print(classification_report(y_test, y_pred,
                             target_names=["Not Poor", "At Risk"]))

print("   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"\n              Predicted")
print(f"              Not Poor   At Risk")
print(f"   Actual  Not Poor  {cm[0][0]:>6}    {cm[0][1]:>6}")
print(f"           At Risk   {cm[1][0]:>6}    {cm[1][1]:>6}")

print("\n   Feature Coefficients (absolute importance):")
coef_pairs = sorted(zip(FEATURE_COLS, abs(model.coef_[0])), key=lambda x: -x[1])
for feat, coef in coef_pairs:
    bar = "=" * int(coef * 20)
    print(f"   {feat:<22} {coef:.4f}  {bar}")

print("=" * 58)
# ▸ Updated label to reflect ₦90,000 threshold
print("\n   Poverty threshold : ₦90,000 income per person")

# ── Save ──────────────────────────────────────────────────────────
joblib.dump(model,  "poverty_model.pkl")
joblib.dump(scaler, "poverty_scaler.pkl")
print("   poverty_model.pkl and poverty_scaler.pkl saved.")