import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)
n = 400

# Generate data
data = pd.DataFrame({
    "income": np.random.randint(20000, 200000, n),
    "household_size": np.random.randint(1, 10, n),
    "education_years": np.random.randint(0, 20, n),
    "employment": np.random.randint(0, 2, n),
    "financial_assets": np.random.randint(0, 500000, n)
})

# Create ratio
data["assets_ratio"] = data["financial_assets"] / data["income"]

# Target variable (poverty logic)
data["poverty"] = (
    (data["income"] < 50000) &
    (data["assets_ratio"] < 1)
).astype(int)

# Features
X = data[["income", "household_size", "education_years", "employment", "assets_ratio"]]
y = data["poverty"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = LogisticRegression()
model.fit(X_scaled, y)

# Save
joblib.dump(model, "poverty_model.pkl")
joblib.dump(scaler, "poverty_scaler.pkl")

print("Model retrained successfully.")