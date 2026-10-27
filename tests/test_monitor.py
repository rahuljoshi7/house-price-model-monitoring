
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

EXPECTED_COLUMNS = {"area", "bedrooms", "location", "price"}


def test_reference_and_current_datasets_exist():
    """Verify reference and current datasets have the required columns."""
    reference_path = DATA_DIR / "reference.csv"
    current_path = DATA_DIR / "current.csv"

    assert reference_path.exists(), "Reference dataset is missing"
    assert current_path.exists(), "Current dataset is missing"

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    assert EXPECTED_COLUMNS.issubset(reference.columns)
    assert EXPECTED_COLUMNS.issubset(current.columns)
    assert len(reference) > 0
    assert len(current) > 0


def test_reference_test_dataset_exists():
    """Verify the held-out reference test dataset."""
    test_path = DATA_DIR / "reference_test.csv"

    assert test_path.exists(), "Reference test dataset is missing"

    reference_test = pd.read_csv(test_path)

    assert EXPECTED_COLUMNS.issubset(reference_test.columns)
    assert len(reference_test) > 0


def test_saved_model_exists_and_loads():
    """Verify the trained model is saved and can be loaded."""
    model_path = MODEL_DIR / "house_price_model.joblib"

    assert model_path.exists(), "Saved model is missing"

    model = joblib.load(model_path)

    assert hasattr(model, "predict")


def test_monitoring_script_generates_report_and_charts():
    """Verify monitoring runs successfully and generates its outputs."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "monitor.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"Monitoring script failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert "Loaded saved model" in result.stdout
    assert "Monitoring completed successfully." in result.stdout

    assert (REPORT_DIR / "monitoring_report.html").exists()
    assert (REPORT_DIR / "data_drift.png").exists()
    assert (REPORT_DIR / "performance_drift.png").exists()


def test_monitoring_history_is_recorded():
    """Verify monitoring history contains the expected metrics."""
    history_path = REPORT_DIR / "monitoring_history.csv"

    assert history_path.exists(), "Monitoring history file is missing"

    history = pd.read_csv(history_path)

    expected_columns = {
        "timestamp_utc",
        "data_drift_detected",
        "reference_mae",
        "current_mae",
        "reference_rmse",
        "current_rmse",
        "reference_r2",
        "current_r2",
        "mae_increase_percent",
        "performance_drift_warning",
    }

    assert expected_columns.issubset(history.columns)
    assert len(history) > 0


def test_report_contains_monitoring_sections():
    """Verify the HTML report contains its tables and embedded charts."""
    report_path = REPORT_DIR / "monitoring_report.html"

    assert report_path.exists()

    report = report_path.read_text(encoding="utf-8")

    assert "Data Drift Results" in report
    assert "Model Performance" in report
    assert "data:image/png;base64," in report
    assert "Historical Monitoring" in report
