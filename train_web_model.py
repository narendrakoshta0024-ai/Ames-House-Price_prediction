
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "train.csv")

# Only features that users enter in the web form
numeric_features = [
    "OverallQual",
    "GrLivArea",
    "YearBuilt",
    "GarageCars",
    "TotalBsmtSF",
    "BedroomAbvGr",
]

categorical_features = [
    "Neighborhood",
    "KitchenQual",
]

features = numeric_features + categorical_features

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])

# Train only on training data
model_pipeline.fit(X_train, y_train)

# Evaluate on unseen test data
predictions = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:    {r2:.4f}")


# SAVE EVALUATION METRICS FOR THE FLASK DASHBOARD
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
}

metrics_path = BASE_DIR / "model_metrics.json"

with open(metrics_path, "w") as file:  
    json.dump(metrics, file, indent=4)

print(f"Metrics saved to: {metrics_path}")


# Save under a NEW name; preserve the existing model
model_path = BASE_DIR / "house_price_model_8features.pkl"
joblib.dump(model_pipeline, model_path)

print(f"\nModel saved to: {model_path}")
