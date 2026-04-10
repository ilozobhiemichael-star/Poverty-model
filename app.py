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
        # Clean function
        def clean(x):
            x = x.replace(",", "").strip()
            if x == "":
                return 0.0
            return float(x)

        # Get inputs
        income = clean(request.form["income"])
        household_size = clean(request.form["household_size"])
        education_years = clean(request.form["education_years"])
        employment = clean(request.form["employment"])
        financial_assets = clean(request.form["financial_assets"])

        # Prevent division by zero
        if household_size == 0:
            household_size = 1

        # New feature (bias fix)
        income_per_person = income / household_size

        # Create feature array (VERY IMPORTANT ORDER)
        features = np.array([[income, household_size, income_per_person,
                              education_years, employment, financial_assets]])

        features_scaled = scaler.transform(features)

        probability = model.predict_proba(features_scaled)[0][1]

        if probability > 0.7:
            prediction = "High Poverty Risk"
        elif probability > 0.4:
            prediction = "Moderate Poverty Risk"
        else:
            prediction = "Low Poverty Risk"

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            error=str(e)
        )
if __name__ == "__main__":
 app.run(debug=True)




