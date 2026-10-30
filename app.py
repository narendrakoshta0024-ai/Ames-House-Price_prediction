from pathlib import Path

import joblib,json
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load the trained ML pipeline
model = joblib.load(BASE_DIR / "house_price_model_8features.pkl")

# Load evaluation metrics
METRICS_PATH = BASE_DIR / "model_metrics.json"

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)


# Features available in our form
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "YearBuilt",
    "GarageCars",
    "TotalBsmtSF",
    "BedroomAbvGr",
    "Neighborhood",
    "KitchenQual",
]


# Allowed numeric ranges from training dataset
NUMERIC_RANGES = {
    "OverallQual": (1, 10),
    "GrLivArea": (334, 5642),
    "YearBuilt": (1872, 2010),
    "GarageCars": (0, 4),
    "TotalBsmtSF": (0, 6110),
    "BedroomAbvGr": (0, 8),
}

# Valid categorical values from training dataset
neighborhood_options = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes",
    "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer",
    "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

kitchen_options = ["Ex", "Fa", "Gd", "TA"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    # Default values shown in the form
    values = {
        "OverallQual": 5,
        "GrLivArea": 1500,
        "YearBuilt": 2000,
        "GarageCars": 2,
        "TotalBsmtSF": 800,
        "BedroomAbvGr": 3,
        "Neighborhood": "CollgCr",
        "KitchenQual": "Gd",
    }

    # Get valid dropdown options from the dataset
    # neighborhood_options = sorted(
    #     df["Neighborhood"].dropna().unique().tolist()
    # )

    # kitchen_options = sorted(
    #     df["KitchenQual"].dropna().unique().tolist()
    # )


    if request.method == "POST":
        try:
            # Read and validate numeric inputs
            for feature, (minimum, maximum) in NUMERIC_RANGES.items():

                raw_value = request.form.get(feature, "").strip()

                if not raw_value:
                    raise ValueError(f"{feature} is required.")

                value = int(raw_value)

                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"{feature} must be between "
                        f"{minimum} and {maximum}."
                    )

                values[feature] = value

            # Read and validate categorical inputs
            neighborhood = request.form.get("Neighborhood", "")
            kitchen_quality = request.form.get("KitchenQual", "")

            if neighborhood not in neighborhood_options:
                raise ValueError("Please select a valid neighborhood.")

            if kitchen_quality not in kitchen_options:
                raise ValueError("Please select a valid kitchen quality.")

            values["Neighborhood"] = neighborhood
            values["KitchenQual"] = kitchen_quality

            # Create DataFrame from user inputs
            new_house = pd.DataFrame([values])

            new_house = new_house[
                [
                    "OverallQual",
                    "GrLivArea",
                    "YearBuilt",
                    "GarageCars",
                    "TotalBsmtSF",
                    "BedroomAbvGr",
                    "Neighborhood",
                    "KitchenQual",
                ]
            ]

            result = model.predict(new_house)
            prediction = f"${result[0]:,.2f}"

        except ValueError as e:
            error = str(e)



        # except (ValueError, KeyError) as e:
        #     error = str(e) or "Please enter valid values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values,
        neighborhood_options=neighborhood_options,
        kitchen_options=kitchen_options,
        metrics = metrics,
        
    )


if __name__ == "__main__":
    app.run(debug=True)
