import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

np.random.seed(42)

# 🔢 Generate 10,000 rows
n = 10000

data = pd.DataFrame({
    "income": np.random.randint(20000, 500000, n),
    "household_size": np.random.randint(1, 10, n),
    "education_years": np.random.randint(0, 20, n),
    "employment": np.random.randint(0, 2, n),
    "financial_assets": np.random.randint(0, 1000000, n)
})

# 🧠 Derived features
data["income_per_person"] = data["income"] / data["household_size"]
data["assets_ratio"] = data["financial_assets"] / (data["income"] + 1)

# 🎯 SMART poverty logic (USES EVERYTHING)
data["poverty"] = (
    (data["income_per_person"] < 25000) |
    (data["employment"] == 0) |
    (data["education_years"] < 6) |
    (data["assets_ratio"] < 0.5)
).astype(int)

# 🎯 Features (VERY IMPORTANT ORDER)
X = data[[
    "income",
    "household_size",
    "income_per_person",
    "education_years",
    "employment",
    "financial_assets"
]]

y = data["poverty"]

# 🔄 Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🤖 Train
model = LogisticRegression()
model.fit(X_scaled, y)

# 💾 Save
joblib.dump(model, "poverty_model.pkl")
joblib.dump(scaler, "poverty_scaler.pkl")

print("✅ Model trained with 10,000 rows")