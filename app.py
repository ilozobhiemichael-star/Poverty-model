from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("poverty_model.pkl")
scaler = joblib.load("poverty_scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html", prediction=None, probability=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Clean ALL numeric inputs (allow commas everywhere)
        def clean(x):
            if x is None or x.strip() == "":
                raise ValueError("Empty input")
            return float(x.replace(",", "").strip())
        income = clean(request.form["income"])
        household_size = clean(request.form["household_size"])
        education_years = clean(request.form["education_years"])
        employment = clean(request.form["employment"])
        financial_assets = clean(request.form["financial_assets"])
        income_per_person = income / household_size
        features = np.array([[income, household_size, income_per_person, education_years, employment, financial_assets]])

        scaled = scaler.transform(features)

        probability = model.predict_proba(scaled)[0][1]
        probability = max(0, min(1, probability))

        prediction = 1 if probability > 0.5 else 0

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except Exception:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            error="Please enter valid numeric values (no letters or symbols)."
        )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)