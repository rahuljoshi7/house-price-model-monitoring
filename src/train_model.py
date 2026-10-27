
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["area", "bedrooms", "location"]
TARGET = "price"

reference = pd.read_csv(DATA_DIR / "reference.csv")

X = reference[FEATURES]
y = reference[TARGET]

# Keep a held-out test set for fair baseline evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Save the trained model and the held-out test data.
joblib.dump(model, MODEL_DIR / "house_price_model.joblib")

reference_test = X_test.copy()
reference_test[TARGET] = y_test
reference_test.to_csv(DATA_DIR / "reference_test.csv", index=False)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Model training completed successfully.")
print(f"Training records: {len(X_train)}")
print(f"Reference test records: {len(X_test)}")
print(f"Reference test MAE: {mae:.2f}")
print(f"Reference test RMSE: {rmse:.2f}")
print(f"Reference test R2: {r2:.3f}")
print(f"Saved model: {MODEL_DIR / 'house_price_model.joblib'}")
print(f"Saved reference test data: {DATA_DIR / 'reference_test.csv'}")
