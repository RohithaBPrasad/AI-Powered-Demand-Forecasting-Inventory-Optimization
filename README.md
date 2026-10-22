# 📦 DemandIQ — AI-Powered Demand Forecasting & Inventory Optimization

**Predict demand. Optimize inventory. Make smarter decisions.**

DemandIQ is a machine learning-powered application that forecasts daily product demand, analyzes sales patterns, estimates prediction ranges, and generates intelligent inventory replenishment recommendations through an interactive Streamlit dashboard.

Built with Python, Scikit-learn, Pandas, Plotly, and SQLite, this project demonstrates how data science can support smarter inventory planning and reduce the risk of stockouts.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
</p>

---

## 🚀 Key Features

- **📈 AI-Powered Demand Forecasting:** Predict future product demand using Random Forest regression and historical sales patterns.
- **📊 Interactive Analytics Dashboard:** Explore sales trends, demand forecasts, inventory alerts, and model performance.
- **📅 Seasonal & Time-Based Analysis:** Engineer lag, rolling-statistic, and calendar features to capture demand patterns.
- **📦 Inventory Optimization:** Calculate safety stock, reorder points, stock coverage, and recommended replenishment quantities.
- **⚠️ Stockout Risk Monitoring:** Identify products that require immediate replenishment or closer monitoring.
- **🎯 Forecast Uncertainty:** Display approximate 80% forecast ranges based on historical holdout errors or historical variability.
- **🧠 Model Evaluation:** Evaluate predictions using chronological holdout testing and Mean Absolute Error (MAE).
- **📂 Flexible Data Upload:** Import sales history through CSV files with automatic data validation and cleaning.
- **🗄️ SQL Data Persistence:** Store cleaned sales records and forecasts in a local SQLite database.
- **⬇️ Exportable Reports:** Download forecasts, cleaned data, and inventory recommendations as CSV files.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data cleaning and numerical analysis |
| Scikit-learn | Random Forest regression and model evaluation |
| Streamlit | Interactive web application |
| Plotly | Interactive charts and visualizations |
| SQLite | Local database storage |
| Git & GitHub | Version control and project hosting |

---

## 🏗️ Project Architecture

```text
                  ┌─────────────────────┐
                  │  Sales Data / CSV   │
                  │  Sample Dataset     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Validation &   │
                  │ Preprocessing       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Feature Engineering │
                  │ Lag + Rolling +     │
                  │ Calendar Features   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Random Forest Model │
                  │ Demand Forecasting  │
                  └──────────┬──────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌──────────────────┐     ┌──────────────────┐
      │ Forecast Ranges  │     │ Inventory Engine │
      │ & Model Metrics  │     │ Reorder Planning │
      └─────────┬────────┘     └─────────┬────────┘
                └────────────┬───────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Streamlit Dashboard │
                  │ SQLite + CSV Export │
                  └─────────────────────┘
```

---

## 📂 Project Structure

```text
demand_forecasting_inventory/
│
├── app.py                 # Streamlit dashboard
├── forecasting.py         # Forecasting model and evaluation
├── inventory.py           # Inventory optimization logic
├── database.py            # SQLite database operations
├── data_generator.py      # Synthetic sales data generator
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
└── data/
    └── sample_sales.csv   # Sample sales dataset
```

The SQLite database is created automatically when the application saves its data.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/demandiq-inventory-forecasting.git
cd demand_forecasting_inventory
```

Replace `YOUR_USERNAME` and the repository URL with your actual GitHub details.

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the VS Code Command Prompt terminal:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

**http://localhost:8501**

You're ready to explore DemandIQ! 🎉

---

## 📊 Dataset Format

DemandIQ includes a synthetic sample dataset containing **8 products and 540 days of generated sales history**.

You can also upload your own CSV dataset.

### Required columns

| Column | Description | Example |
|---|---|---|
| `date` | Sales transaction date | `2026-01-01` |
| `product_id` | Product or SKU identifier | `SKU-001` |
| `sales` | Units sold on that date | `24` |

### Optional columns

| Column | Description |
|---|---|
| `inventory` | Recorded on-hand stock |
| `unit_cost` | Cost per product unit |
| `lead_time_days` | Supplier lead time in days |
| `promotion` | `1` if promoted; otherwise `0` |

### Example CSV

```csv
date,product_id,sales,inventory,unit_cost,lead_time_days,promotion
2026-01-01,SKU-001,24,150,25.50,7,0
2026-01-02,SKU-001,28,122,25.50,7,1
2026-01-03,SKU-001,19,103,25.50,7,0
2026-01-01,SKU-002,15,100,18.00,5,0
2026-01-02,SKU-002,18,82,18.00,5,0
```

Use actual, consistently recorded inventory and cost data for meaningful replenishment estimates.

---

## 🧠 How the Model Works

1. **Data preprocessing:** Validate records, convert dates, handle invalid values, and aggregate duplicate product-date entries.
2. **Feature engineering:** Generate historical lags, rolling averages, rolling standard deviations, and calendar features.
3. **Model training:** Train a Random Forest regression model for each product with sufficient historical data.
4. **Time-aware evaluation:** Use a chronological holdout to estimate Mean Absolute Error (MAE).
5. **Demand forecasting:** Generate daily predictions recursively over the selected forecast horizon.
6. **Uncertainty estimation:** Calculate approximate forecast ranges using holdout residual quantiles or historical variability.
7. **Inventory optimization:** Estimate safety stock, reorder points, days of cover, and suggested order quantities.
8. **Decision support:** Present the results in an interactive dashboard with downloadable reports.

---

## 📦 Inventory Optimization Logic

DemandIQ uses transparent inventory planning formulas.

**Safety Stock**

\[
SS = z \times \sigma_d \times \sqrt{L}
\]

Where:

- \(z\) = service-level factor
- \(\sigma_d\) = estimated standard deviation of daily demand
- \(L\) = supplier lead time in days

**Reorder Point**

\[
ROP = \bar{d} \times L + SS
\]

Where \(\bar{d}\) is average daily demand.

**Suggested Order Quantity**

The application estimates a target stock level for the supplier lead time plus the inventory review period, then calculates the positive difference between the target and current stock.

These formulas are simplified planning estimates; more advanced inventory policies may be needed for variable lead times, intermittent demand, supplier constraints, and changing demand distributions.

---

## 📉 Model Evaluation

The project uses **Mean Absolute Error (MAE)** on a chronological holdout to measure the average absolute difference between actual and predicted daily demand.

Lower MAE generally indicates better predictions for the evaluated dataset.

For stronger production-grade evaluation, consider adding:

- Rolling-origin time-series cross-validation
- Seasonal-naive and other baseline comparisons
- Weighted Absolute Percentage Error (WAPE)
- Mean Absolute Scaled Error (MASE)
- Forecast bias and prediction-interval coverage
- Demand drift monitoring

Forecast ranges are approximate estimates, not guaranteed confidence or prediction bounds.

---

## 🎯 Use Cases

- 🛒 Retail demand planning
- 🏬 Inventory and warehouse management
- 📦 Stock replenishment planning
- 📊 Sales trend analysis
- 🚚 Supply chain decision support
- 🧪 Data Science experimentation and portfolio demonstrations

---

## 🔮 Future Enhancements

- [ ] XGBoost and LightGBM model comparison
- [ ] Dedicated time-series forecasting models
- [ ] Holiday, price, and promotion-aware forecasting
- [ ] Rolling-origin backtesting and model selection
- [ ] Intermittent-demand forecasting methods
- [ ] Supplier minimum order quantities and pack-size constraints
- [ ] Automated scheduled retraining
- [ ] Forecast drift monitoring
- [ ] Cloud database integration
- [ ] User authentication and role-based access
- [ ] Cloud deployment and CI/CD pipeline

---

## ⚠️ Limitations

- The bundled dataset is synthetic and does not represent actual business performance.
- Missing product-date records are interpreted as zero sales during forecasting.
- Prediction intervals are approximate and require further calibration.
- Inventory recommendations depend on the quality of stock, cost, and lead-time inputs.
- The current implementation is an educational and portfolio baseline, not a validated autonomous purchasing system.

Validate the model against real historical data before using its outputs for operational or purchasing decisions.

---

## 👨‍💻 Author

**ROHITHA B PRASAD**

Data Science | Machine Learning | Predictive Analytics


---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub!

**DemandIQ — Turning historical sales data into smarter inventory decisions.**
