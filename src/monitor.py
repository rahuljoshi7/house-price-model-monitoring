
from pathlib import Path
from datetime import datetime, timezone
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.stats import ks_2samp, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["area", "bedrooms", "location"]
TARGET = "price"

reference = pd.read_csv(DATA_DIR / "reference.csv")
current = pd.read_csv(DATA_DIR / "current.csv")

# 1. Detect data drift
drift_results = []

for feature in ["area", "bedrooms"]:
    statistic, p_value = ks_2samp(reference[feature], current[feature])
    drift_results.append({
        "Feature": feature,
        "Test": "KS test",
        "Statistic": statistic,
        "P-value": p_value,
        "Drift detected": bool(p_value < 0.05)
    })

ref_counts = reference["location"].value_counts()
cur_counts = current["location"].value_counts()
categories = sorted(set(ref_counts.index) | set(cur_counts.index))

contingency = pd.DataFrame({
    "Reference": ref_counts.reindex(categories, fill_value=0),
    "Current": cur_counts.reindex(categories, fill_value=0)
})

chi2, p_value, _, _ = chi2_contingency(contingency.T)

drift_results.append({
    "Feature": "location",
    "Test": "Chi-square test",
    "Statistic": chi2,
    "P-value": p_value,
    "Drift detected": bool(p_value < 0.05)
})

drift_df = pd.DataFrame(drift_results)
overall_data_drift = bool(drift_df["Drift detected"].any())



# 2. Load the previously trained model
model_path = MODEL_DIR / "house_price_model.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Trained model not found. Run: python src/train_model.py"
    )

model = joblib.load(model_path)
print(f"Loaded saved model: {model_path}")

# 3. Load the held-out reference test set
reference_test = pd.read_csv(DATA_DIR / "reference_test.csv")

X_reference_test = reference_test[FEATURES]
y_reference_test = reference_test[TARGET]

# 4. Evaluate the saved model on reference and current data
reference_predictions = model.predict(X_reference_test)
current_predictions = model.predict(current[FEATURES])

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }

reference_metrics = calculate_metrics(
    y_reference_test, reference_predictions
)
current_metrics = calculate_metrics(
    current[TARGET], current_predictions
)

performance_df = pd.DataFrame([
    {"Dataset": "Reference test", **reference_metrics},
    {"Dataset": "Current production", **current_metrics}
])

mae_increase = (
    (current_metrics["MAE"] - reference_metrics["MAE"])
    / max(reference_metrics["MAE"], 1e-9)
) * 100

performance_drift = bool(mae_increase > 20)

# 3. Persist the trained model
model_path = MODEL_DIR / "house_price_model.joblib"
joblib.dump(model, model_path)

# 4. Evaluate reference and current performance
reference_test = pd.read_csv(DATA_DIR / "reference_test.csv")

X_reference_test = reference_test[FEATURES]
y_reference_test = reference_test[TARGET]

reference_predictions = model.predict(X_reference_test)
current_predictions = model.predict(current[FEATURES])

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }

reference_metrics = calculate_metrics(
    y_reference_test, reference_predictions
)
current_metrics = calculate_metrics(current[TARGET], current_predictions)

performance_df = pd.DataFrame([
    {"Dataset": "Reference test", **reference_metrics},
    {"Dataset": "Current production", **current_metrics}
])

mae_increase = (
    (current_metrics["MAE"] - reference_metrics["MAE"])
    / max(reference_metrics["MAE"], 1e-9)
) * 100

performance_drift = bool(mae_increase > 20)

# 5. Save historical monitoring results (append, do not overwrite)
history_path = REPORT_DIR / "monitoring_history.csv"

history_record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reference_records": len(reference),
    "current_records": len(current),
    "data_drift_detected": overall_data_drift,
    "area_p_value": float(drift_df.loc[drift_df["Feature"] == "area", "P-value"].iloc[0]),
    "bedrooms_p_value": float(drift_df.loc[drift_df["Feature"] == "bedrooms", "P-value"].iloc[0]),
    "location_p_value": float(drift_df.loc[drift_df["Feature"] == "location", "P-value"].iloc[0]),
    "reference_mae": reference_metrics["MAE"],
    "current_mae": current_metrics["MAE"],
    "reference_rmse": reference_metrics["RMSE"],
    "current_rmse": current_metrics["RMSE"],
    "reference_r2": reference_metrics["R2"],
    "current_r2": current_metrics["R2"],
    "mae_increase_percent": float(mae_increase),
    "performance_drift_warning": performance_drift
}

history_df = pd.DataFrame([history_record])

if history_path.exists():
    history_df.to_csv(history_path, mode="a", header=False, index=False)
else:
    history_df.to_csv(history_path, index=False)

# 6. Generate data drift visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, feature in zip(axes, FEATURES):
    if feature == "location":
        ref_dist = reference[feature].value_counts(normalize=True)
        cur_dist = current[feature].value_counts(normalize=True)
        distribution = pd.DataFrame({
            "Reference": ref_dist,
            "Current": cur_dist
        }).fillna(0)
        distribution.plot(kind="bar", ax=ax)
        ax.set_ylabel("Proportion")
    else:
        ax.hist(
            reference[feature], bins=25, alpha=0.6,
            label="Reference", density=True
        )
        ax.hist(
            current[feature], bins=25, alpha=0.6,
            label="Current", density=True
        )
        ax.set_ylabel("Density")
        ax.legend()

    ax.set_title(f"{feature.title()} distribution")
    ax.set_xlabel(feature.title())

fig.tight_layout()
data_plot = REPORT_DIR / "data_drift.png"
fig.savefig(data_plot, dpi=150, bbox_inches="tight")
plt.close(fig)

# 7. Generate performance visualization
fig, ax = plt.subplots(figsize=(8, 5))
performance_df.set_index("Dataset")[["MAE", "RMSE"]].plot(kind="bar", ax=ax)
ax.set_title("Model Error: Reference vs Current")
ax.set_ylabel("Error (price units)")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()

performance_plot = REPORT_DIR / "performance_drift.png"
fig.savefig(performance_plot, dpi=150, bbox_inches="tight")
plt.close(fig)

# 8. Generate HTML monitoring report with embedded charts
def image_as_base64(path):
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

drift_table = drift_df.to_html(
    index=False, float_format=lambda value: f"{value:.4f}"
)
performance_table = performance_df.to_html(
    index=False, float_format=lambda value: f"{value:.2f}"
)

data_image = image_as_base64(data_plot)
performance_image = image_as_base64(performance_plot)

data_status = (
    "DRIFT DETECTED" if overall_data_drift
    else "No significant drift detected"
)
performance_status = (
    "PERFORMANCE DRIFT WARNING" if performance_drift
    else "No major performance degradation detected"
)

html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>House Price Model Monitoring Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px;
       margin: 30px auto; padding: 0 20px; color: #222; }}
h1, h2 {{ color: #19324d; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #edf2f7; }}
img {{ max-width: 100%; height: auto; margin: 12px 0 28px; }}
.status {{ padding: 12px; background: #f2f4f7; margin: 10px 0; }}
.note {{ color: #555; font-size: 14px; }}
</style>
</head>
<body>
<h1>House Price Prediction — Monitoring Report</h1>
<p>Reference records: {len(reference)} |
Current records: {len(current)}</p>

<h2>1. Data Drift Results</h2>
<div class="status"><strong>Overall result:</strong> {data_status}</div>
{drift_table}
<img src="{data_image}" alt="Feature distribution comparison">

<h2>2. Model Performance / Concept Drift Monitoring</h2>
<div class="status"><strong>Overall result:</strong> {performance_status}</div>
<p>Current MAE change compared with reference test: {mae_increase:.2f}%</p>
{performance_table}
<img src="{performance_image}" alt="Model error comparison">

<h2>3. Historical Monitoring</h2>
<p>Each run is appended to reports/monitoring_history.csv.
The saved model is models/house_price_model.joblib.</p>

<h2>4. Interpretation and Limitations</h2>
<ul>
<li>Data drift means input feature distributions have changed.</li>
<li>MAE and RMSE measure prediction errors; lower values are better.</li>
<li>R² values closer to 1 generally indicate a better fit.</li>
<li>A performance warning is raised when current MAE is more than
20% higher than reference test MAE.</li>
</ul>
<p class="note">This assignment uses synthetic data. Real production
performance monitoring requires actual house prices to become available.
Performance degradation can indicate concept drift but does not prove it
by itself. The model is saved for reuse; this script retrains the baseline
model on each run for reproducible assignment evaluation.</p>
</body>
</html>
"""

report_path = REPORT_DIR / "monitoring_report.html"
report_path.write_text(html, encoding="utf-8")

# 9. Print summary
print("\n========== DATA DRIFT RESULTS ==========")
print(drift_df.to_string(index=False))
print("\n========== MODEL PERFORMANCE ==========")
print(performance_df.to_string(index=False))
print(f"\nCurrent MAE change: {mae_increase:.2f}%")
print(f"Data drift: {data_status}")
print(f"Performance monitoring: {performance_status}")
print(f"\nSaved model: {model_path}")
print(f"Monitoring history: {history_path}")
print(f"HTML report: {report_path}")
print(f"Data drift chart: {data_plot}")
print(f"Performance chart: {performance_plot}")
print("\nMonitoring completed successfully.")
