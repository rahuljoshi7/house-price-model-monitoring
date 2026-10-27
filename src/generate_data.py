from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


def generate_houses(n, drift=False):
    if not drift:
        area = rng.normal(1500, 350, n).clip(500, 3500)
        bedrooms = rng.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
        location = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.5, 0.3, 0.2])
        noise = rng.normal(0, 20000, n)
        location_effect = pd.Series(location).map(
            {"Urban": 60000, "Suburban": 30000, "Rural": 0}
        ).to_numpy()
        price = 50000 + area * 150 + bedrooms * 20000 + location_effect + noise
    else:
        # Simulate changed production inputs and a changed price relationship.
        area = rng.normal(2100, 500, n).clip(600, 4500)
        bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.3, 0.3, 0.2])
        location = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.25, 0.35, 0.4])
        noise = rng.normal(0, 35000, n)
        location_effect = pd.Series(location).map(
            {"Urban": 30000, "Suburban": 50000, "Rural": 10000}
        ).to_numpy()
        # Concept drift: the relationship between area and price has changed.
        price = 70000 + area * 115 + bedrooms * 12000 + location_effect + noise

    return pd.DataFrame({
        "area": np.round(area, 1),
        "bedrooms": bedrooms,
        "location": location,
        "price": np.round(price, 2),
    })


reference = generate_houses(1000, drift=False)
current = generate_houses(1000, drift=True)

reference.to_csv(DATA_DIR / "reference.csv", index=False)
current.to_csv(DATA_DIR / "current.csv", index=False)

print("Synthetic house-price datasets created successfully.")
print(f"Reference records: {len(reference)}")
print(f"Current records: {len(current)}")
print("\nReference sample:")
print(reference.head())
print("\nCurrent sample:")
print(current.head())