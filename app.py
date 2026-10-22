from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import make_sample_data
from forecasting import validate_sales_data, forecast_all
from inventory import build_inventory_recommendations
import database

st.set_page_config(page_title="DemandIQ | Forecast & Inventory", page_icon="📦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#10243a,#1e5268); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0 0 .3rem 0;}
.hero p {color: #d9e8f0; margin: 0;}
.metric-note {color: #64748b; font-size: .85rem;}
</style>
<div class="hero">
<h1>DemandIQ</h1>
<p>AI-powered demand forecasting · Inventory optimization · Decision intelligence</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def sample_data():
    return make_sample_data(days=540, products=8)

def money(value):
    return f"₹{value:,.0f}"

with st.sidebar:
    st.header("Data workspace")
    source = st.radio("Choose data source", ["Built-in sample data", "Upload your CSV"], index=0)
    uploaded = None
    if source == "Upload your CSV":
        uploaded = st.file_uploader("Upload sales history (.csv)", type=["csv"])
    st.divider()
    st.subheader("Forecast settings")
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7)
    service_level = st.select_slider("Target service level", options=["90%", "95%", "97.5%", "99%"], value="95%")
    z_map = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}
    review_period = st.slider("Inventory review period (days)", 1, 30, 7)
    run = st.button("Run forecasting pipeline", type="primary", use_container_width=True)
    st.caption("CSV required columns: date, product_id, sales. Optional: inventory, unit_cost, lead_time_days, promotion.")

if source == "Built-in sample data":
    raw = sample_data()
elif uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    st.info("Upload a CSV file or select Built-in sample data.")
    st.stop()

try:
    sales_df = validate_sales_data(raw)
except Exception as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()

# Pipeline recalculates when requested, or when no results are available in this session.
input_signature = (source, getattr(uploaded, "name", None), len(sales_df), str(sales_df["date"].max()), horizon, service_level, review_period)
if "pipeline_signature" not in st.session_state:
    st.session_state.pipeline_signature = None
if run or "forecast_df" not in st.session_state or st.session_state.pipeline_signature != input_signature:
    with st.spinner("Training product-level models and generating inventory recommendations..."):
        forecast_df, metrics_df = forecast_all(sales_df, horizon)
        inventory_df = build_inventory_recommendations(sales_df, forecast_df, z_map[service_level], review_period)
        st.session_state.forecast_df = forecast_df
        st.session_state.metrics_df = metrics_df
        st.session_state.inventory_df = inventory_df
        st.session_state.pipeline_signature = input_signature
        try:
            database.save_sales(sales_df)
            database.save_forecasts(forecast_df)
        except Exception as exc:
            st.warning(f"Forecasts are ready, but SQLite persistence failed: {exc}")

forecast_df = st.session_state.forecast_df.copy()
metrics_df = st.session_state.metrics_df.copy()
inventory_df = st.session_state.inventory_df.copy()

st.caption(f"Data window: {sales_df['date'].min().date()} to {sales_df['date'].max().date()} · {sales_df['product_id'].nunique()} products · {len(sales_df):,} product-day records")

total_products = sales_df["product_id"].nunique()
reorder_count = int((inventory_df["status"] == "REORDER NOW").sum())
watch_count = int((inventory_df["status"] == "WATCH").sum())
order_value = float(inventory_df["estimated_order_value"].sum())
total_forecast = float(forecast_df["forecast"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products tracked", f"{total_products:,}")
c2.metric("Forecast units", f"{total_forecast:,.0f}", f"next {horizon} days")
c3.metric("Reorder alerts", f"{reorder_count}", f"{watch_count} products to watch")
c4.metric("Suggested order value", money(order_value))

tabs = st.tabs(["Overview", "Demand forecast", "Inventory planner", "Model performance", "Data & exports"])

with tabs[0]:
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Historical sales and forecast")
        product_options = sorted(sales_df["product_id"].unique())
        selected = st.selectbox("Product", product_options, key="overview_product")
        hist = sales_df[sales_df["product_id"] == selected].groupby("date", as_index=False)["sales"].sum().tail(120)
        fut = forecast_df[forecast_df["product_id"] == selected]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="Actual sales", mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["upper_80"], name="Upper range (80%)", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["lower_80"], name="Approx. 80% forecast range", fill="tonexty", mode="lines", line=dict(width=0), opacity=0.22))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast"], name="Forecast", mode="lines", line=dict(width=3, dash="dash")))
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=25,b=10), xaxis_title="", yaxis_title="Units", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Inventory status")
        status_counts = inventory_df["status"].value_counts().reindex(["REORDER NOW", "WATCH", "HEALTHY"], fill_value=0).reset_index()
        status_counts.columns = ["Status", "Products"]
        st.plotly_chart(px.pie(status_counts, names="Status", values="Products", hole=0.58), use_container_width=True)
        st.markdown("**Highest-priority products**")
        st.dataframe(inventory_df.sort_values("recommended_order_qty", ascending=False)[["product_id","status","current_stock","reorder_point","recommended_order_qty"]].head(5), use_container_width=True, hide_index=True)
    st.info("Forecast intervals are approximate and should be validated against your business's historical forecast errors before operational use.")

with tabs[1]:
    st.subheader("Demand forecast explorer")
    fc_product = st.selectbox("Select product to inspect", sorted(forecast_df["product_id"].unique()), key="forecast_product")
    fc = forecast_df[forecast_df["product_id"] == fc_product].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper_80"], name="Upper estimate", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower_80"], name="Approx. 80% range", fill="tonexty", line=dict(width=0), opacity=0.2))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Expected demand", mode="lines+markers"))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Forecast units", margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fc, use_container_width=True, hide_index=True)
    st.download_button("Download this product forecast", fc.to_csv(index=False).encode("utf-8"), file_name=f"{fc_product}_forecast.csv", mime="text/csv")

with tabs[2]:
    st.subheader("Inventory recommendations")
    st.write("Reorder point = expected demand during supplier lead time + safety stock. Order quantity targets lead time plus the review period.")
    status_filter = st.multiselect("Filter by status", ["REORDER NOW", "WATCH", "HEALTHY"], default=["REORDER NOW", "WATCH", "HEALTHY"])
    view = inventory_df[inventory_df["status"].isin(status_filter)].copy()
    st.dataframe(view, use_container_width=True, hide_index=True, column_config={
        "estimated_order_value": st.column_config.NumberColumn("Estimated order value (₹)", format="₹%.2f"),
        "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
        "recommended_order_qty": st.column_config.NumberColumn("Recommended order qty", format="%d"),
    })
    st.download_button("Download inventory recommendations", inventory_df.to_csv(index=False).encode("utf-8"), file_name="inventory_recommendations.csv", mime="text/csv")
    st.caption("The suggested order value is a planning estimate based on the unit_cost column; it does not include tax, freight, minimum order quantities, or supplier constraints.")

with tabs[3]:
    st.subheader("Forecast model diagnostics")
    st.write("Evaluation uses a chronological holdout rather than randomly shuffling time-series records.")
    display_metrics = metrics_df.copy()
    display_metrics["mae"] = display_metrics["mae"].round(2)
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    valid_mae = metrics_df.dropna(subset=["mae"])
    if not valid_mae.empty:
        st.metric("Mean product-level holdout MAE", f"{valid_mae['mae'].mean():.2f} units/day")
    st.warning("This baseline project reports MAE and approximate forecast ranges. For production, add rolling-origin backtesting, seasonal-naive comparisons, WAPE/MASE, drift monitoring, and promotion/price/holiday features.")

with tabs[4]:
    st.subheader("Data quality summary")
    q1, q2, q3 = st.columns(3)
    q1.metric("Valid records", f"{len(sales_df):,}")
    q2.metric("Missing values after cleaning", f"{int(sales_df.isna().sum().sum()):,}")
    q3.metric("Duplicate product-date keys", f"{int(sales_df.duplicated(['date','product_id']).sum()):,}")
    st.dataframe(sales_df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download cleaned sales data", sales_df.to_csv(index=False).encode("utf-8"), file_name="cleaned_sales_data.csv", mime="text/csv")
    st.download_button("Download all forecasts", forecast_df.to_csv(index=False).encode("utf-8"), file_name="all_product_forecasts.csv", mime="text/csv")
    st.markdown("**SQLite database**")
    st.code("data/inventory_forecasting.db\nTables: sales_history, demand_forecasts", language="text")

with st.expander("Data schema and assumptions"):
    st.markdown("""
    **Required CSV columns**
    - `date`: calendar date such as `2026-01-31`
    - `product_id`: product/SKU identifier
    - `sales`: units sold on that date (non-negative number)

    **Optional columns**
    - `inventory`: on-hand units recorded for that product/date
    - `unit_cost`: estimated cost per unit, in your chosen currency
    - `lead_time_days`: supplier lead time in days
    - `promotion`: 1 if promoted, otherwise 0

    **Important assumptions**
    - Missing product/date rows are interpreted as zero sales during daily forecasting.
    - Current inventory is taken from the latest row for each product.
    - Prediction intervals are approximate, not guaranteed confidence bounds.
    - Sample data is synthetic and must not be represented as real business performance.
    """)
