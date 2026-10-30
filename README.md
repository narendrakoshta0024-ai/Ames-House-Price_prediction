# Ames House Price Prediction

A Machine Learning web application that predicts house prices using property-related features.

## Project Overview

This project uses a Random Forest Regression model to estimate house prices based on 8 selected features.

Users can enter property details through a Flask web interface and receive an estimated house price.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Flask
- HTML
- CSS
- Joblib

## Machine Learning Model

**Algorithm:** Random Forest Regressor

**Selected Features:**
- OverallQual
- GrLivArea
- YearBuilt
- GarageCars
- TotalBsmtSF
- BedroomAbvGr
- Neighborhood
- KitchenQual

## Model Performance

| Metric | Result |
|---|---:|
| MAE | $18,918.63 |
| RMSE | $29,388.41 |
| R² Score | 0.8803 |

These metrics are calculated on the test dataset. Predictions are estimates, not guaranteed market prices.

## Project Structure

```text
Ames-House-Price-Prediction/
├── data/
├── templates/
├── static/
├── app.py
├── train_web_model.py
├── house_price_model_8features.pkl
├── model_metrics.json
├── requirements.txt
└── README.md
```

## How to Run the Project

1. Clone the repository.
2. Open the project folder.
3. Create and activate a Python virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Flask application:

```bash
python app.py
```

6. Open the local URL displayed in your terminal.

## Disclaimer

This project is developed for machine learning learning and demonstration purposes. Predicted prices are estimates and should not be treated as actual property valuations.