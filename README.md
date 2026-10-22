# DemandIQ — AI Demand Forecasting & Inventory Optimization

A ready-to-run portfolio project that forecasts daily product demand, estimates approximate prediction ranges, and generates transparent inventory replenishment recommendations.

## Features

- Streamlit interactive dashboard
- Synthetic sample dataset (8 products, 540 days)
- CSV upload and data validation
- Product-level Random Forest forecasting with lag, rolling, and calendar features
- Chronological holdout MAE evaluation
- Approximate 80% prediction intervals from holdout residuals (fallback to historical variation)
- Reorder point, safety stock, days of cover, and suggested order quantity
- SQLite persistence for cleaned sales and forecast tables
- CSV downloads for forecasts and inventory recommendations

## Requirements

- Python 3.10–3.12 recommended
- Internet connection for installing dependencies
- Windows, macOS, or Linux

## Run on Windows (VS Code)

1. Extract the ZIP.
2. Open the extracted `demand_forecasting_inventory` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal:

   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Launch the app:

   ```powershell
   streamlit run app.py
   ```

7. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Minimum required columns:

```csv
date,product_id,sales
2026-01-01,SKU-001,24
2026-01-02,SKU-001,28
2026-01-01,SKU-002,15
```

Optional columns:

- `inventory`: on-hand stock
- `unit_cost`: unit cost in the currency you use
- `lead_time_days`: supplier lead time
- `promotion`: 1 for promotion, otherwise 0

The app adds defaults for optional fields if they are absent. Inventory recommendations are meaningful only when the uploaded inventory values and costs reflect your actual business.

## Project structure

```text
demand_forecasting_inventory/
├── app.py
├── forecasting.py
├── inventory.py
├── database.py
├── data_generator.py
├── requirements.txt
├── README.md
└── data/
    └── sample_sales.csv
```

`data/inventory_forecasting.db` is created when the app saves data.

## How the model works

1. Validate and aggregate rows by product/date.
2. Build daily sales series, filling absent dates with zero sales.
3. Generate lag features (1, 7, 14, 28 days), rolling means/standard deviation, and calendar features.
4. Evaluate a model on a chronological holdout, then fit the final Random Forest.
5. Predict recursively across the selected horizon.
6. Estimate an approximate interval from held-out residual quantiles.
7. Calculate inventory safety stock, reorder points, and order quantities.

## Important limitations

This is a complete educational/portfolio baseline, not a validated production planning system. Before using it to place real orders, validate it on your own data and add rolling-origin backtesting, appropriate intermittent-demand methods, supplier minimum order quantities, pack sizes, promotions, price and holiday features, and business-specific stock policies. A missing date is treated as zero sales, which is not suitable if missing records mean missing data rather than no sales. Synthetic data is for demonstration only.
