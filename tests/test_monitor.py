
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def test_reference_and_current_datasets_exist():
    """Both datasets should exist and contain the expected columns."""
    reference_path = PROJECT_ROOT / "data" / "reference.csv"
    current_path = PROJECT_ROOT / "data" / "current.csv"

    assert reference_path.exists()
    assert current_path.exists()

    expected_columns = {"area", "bedrooms", "location", "price"}
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    assert expected_columns.issubset(reference.columns)
    assert expected_columns.issubset(current.columns)
    assert len(reference) > 0
    assert len(current) > 0


def test_monitoring_script_generates_report_and_charts():
    """Running the monitor should generate the HTML report and charts."""
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

    assert (REPORT_DIR / "monitoring_report.html").exists()
    assert (REPORT_DIR / "data_drift.png").exists()
    assert (REPORT_DIR / "performance_drift.png").exists()


def test_report_contains_monitoring_sections():
    """The generated report should include both monitoring sections."""
    report_path = REPORT_DIR / "monitoring_report.html"

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")

    assert "Data Drift Results" in report
    assert "Model Performance" in report
    assert "data:image/png;base64," in report
