from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model  = joblib.load("poverty_model.pkl")
scaler = joblib.load("poverty_scaler.pkl")


def clean(value):
    """Strip commas and whitespace, return float. Empty = 0.0"""
    return float(value.replace(",", "").strip() or 0.0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        income           = clean(request.form["income"])
        household_size   = max(clean(request.form["household_size"]), 1)
        education_years  = clean(request.form["education_years"])
        employment       = clean(request.form["employment"])
        financial_assets = clean(request.form["financial_assets"])

        income_per_person = income / household_size

        features = np.array([[
            income, household_size, income_per_person,
            education_years, employment, financial_assets
        ]])

        features_scaled = scaler.transform(features)
        probability     = model.predict_proba(features_scaled)[0][1]
        prob_pct        = round(probability * 100, 1)

        if probability > 0.7:
            level    = "high"
            verdict  = "High Poverty Risk"
            message  = "This household shows significant vulnerability indicators. Immediate intervention is recommended."
        elif probability > 0.4:
            level    = "moderate"
            verdict  = "Moderate Poverty Risk"
            message  = "This household shows some vulnerability. Targeted social support programs may be beneficial."
        else:
            level    = "low"
            verdict  = "Low Poverty Risk"
            message  = "This household shows relatively stable economic indicators."

        # Individual risk flags
        flags = []
        if income_per_person < 25000:
            flags.append("Income per capita below poverty threshold")
        if employment == 0:
            flags.append("No active employment reported")
        if education_years < 6:
            flags.append("Education below primary level")
        assets_ratio = financial_assets / (income + 1)
        if assets_ratio < 0.5:
            flags.append("Low financial asset buffer")

        return render_template(
            "index.html",
            verdict=verdict,
            level=level,
            prob_pct=prob_pct,
            message=message,
            flags=flags,
            form_data=request.form,
        )

    except Exception as e:
        return render_template("index.html", error=str(e), form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True)