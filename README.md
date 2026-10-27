# House Price Prediction Model Monitoring

## Project Overview
This project monitors a house price prediction model to identify data drift and changes in model performance. It generates a monitoring report with visualizations.

## Objectives
- Detect data drift in area, bedrooms, and location.
- Monitor prediction performance using MAE, RMSE, and R².
- Compare reference data with current production data.
- Generate an HTML report and visualization charts.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Matplotlib

## Project Structure
```text
House-Price-Monitoring/
├── data/
│   ├── reference.csv
│   └── current.csv
├── reports/
│   ├── data_drift.png
│   ├── performance_drift.png
│   └── monitoring_report.html
├── src/
│   ├── generate_data.py
│   └── monitor.py
├── requirements.txt
└── README.md
```

## Setup and Execution

### 1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Generate datasets
```powershell
python src\generate_data.py
```

### 4. Run model monitoring
```powershell
python src\monitor.py
```

### 5. Open the monitoring report
```powershell
Start-Process reports\monitoring_report.html
```

## Monitoring Methodology
- Numerical features: Kolmogorov-Smirnov test.
- Categorical feature: Chi-square test.
- Model performance: MAE, RMSE, and R².
- Performance warning: MAE increase greater than 20% compared with the reference test set.

## Results
The generated synthetic datasets demonstrate changes in feature distributions and model performance. The monitoring report contains test results, metrics, and visualizations.

## Limitation
Actual production performance monitoring requires real house prices to become available. A performance decline can indicate possible concept drift, but it does not establish concept drift by itself.

## Data Disclaimer
The datasets are synthetic and generated for educational demonstration. They are not actual real-estate market data.
