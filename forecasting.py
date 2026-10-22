from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE_COL = "date"
REQUIRED_COLUMNS = {"date", "product_id", "sales"}

def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "product_id", "sales"])
    df = df[(df["product_id"] != "") & (df["sales"] >= 0)]
    if df.empty:
        raise ValueError("No valid rows remain. Dates must be valid and sales must be non-negative.")
    for col, default in [("inventory", 100), ("unit_cost", 10.0), ("lead_time_days", 7), ("promotion", 0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["inventory"] = df["inventory"].clip(lower=0)
    df["unit_cost"] = df["unit_cost"].clip(lower=0)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=1).astype(int)
    df["promotion"] = df["promotion"].clip(0, 1).astype(int)
    # Aggregate duplicate product/date rows, retaining latest available inventory metadata.
    agg = df.groupby(["date", "product_id"], as_index=False).agg(
        sales=("sales", "sum"),
        inventory=("inventory", "last"),
        unit_cost=("unit_cost", "last"),
        lead_time_days=("lead_time_days", "last"),
        promotion=("promotion", "max"),
    )
    return agg.sort_values(["product_id", "date"]).reset_index(drop=True)

def make_features(series: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = series.astype(float).reset_index(drop=True)
    n = len(values)
    rows = []
    for i, date in enumerate(dates):
        def lag(k):
            idx = i-k
            return float(values.iloc[idx]) if idx >= 0 else np.nan
        past7 = values.iloc[max(0, i-7):i]
        past14 = values.iloc[max(0, i-14):i]
        past28 = values.iloc[max(0, i-28):i]
        rows.append({
            "dow": date.dayofweek, "day": date.day, "month": date.month,
            "week": int(date.isocalendar().week), "quarter": date.quarter,
            "is_weekend": int(date.dayofweek >= 5),
            "lag_1": lag(1), "lag_7": lag(7), "lag_14": lag(14), "lag_28": lag(28),
            "roll_mean_7": float(past7.mean()) if len(past7) else np.nan,
            "roll_mean_14": float(past14.mean()) if len(past14) else np.nan,
            "roll_mean_28": float(past28.mean()) if len(past28) else np.nan,
            "roll_std_7": float(past7.std(ddof=0)) if len(past7) else np.nan,
        })
    return pd.DataFrame(rows, index=dates)

def _feature_row(history: list[float], date: pd.Timestamp) -> dict:
    def lag(k):
        return float(history[-k]) if len(history) >= k else np.nan
    def mean(k):
        return float(np.mean(history[-k:])) if history else np.nan
    def std(k):
        return float(np.std(history[-k:])) if history else np.nan
    return {
        "dow": date.dayofweek, "day": date.day, "month": date.month,
        "week": int(date.isocalendar().week), "quarter": date.quarter,
        "is_weekend": int(date.dayofweek >= 5),
        "lag_1": lag(1), "lag_7": lag(7), "lag_14": lag(14), "lag_28": lag(28),
        "roll_mean_7": mean(7), "roll_mean_14": mean(14), "roll_mean_28": mean(28),
        "roll_std_7": std(7),
    }

def forecast_product(history_df: pd.DataFrame, horizon: int = 30) -> tuple[pd.DataFrame, dict]:
    """Train a time-aware random forest model and recursively forecast one product."""
    history_df = history_df.sort_values("date").copy()
    # Fill missing calendar days as zero sales; this is appropriate when absent dates mean no sales.
    full_dates = pd.date_range(history_df["date"].min(), history_df["date"].max(), freq="D")
    daily = history_df.set_index("date")["sales"].groupby(level=0).sum().reindex(full_dates, fill_value=0).astype(float)
    daily.index.name = "date"
    product = str(history_df["product_id"].iloc[0])
    if len(daily) < 5:
        avg = float(daily.mean()) if len(daily) else 0.0
        future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        preds = np.repeat(max(0, avg), horizon)
        method = "Short-history average baseline"
        mae = np.nan
    else:
        features = make_features(daily, daily.index)
        target = daily.values
        valid = features.notna().all(axis=1).values
        X = features.loc[valid]
        y = target[valid]
        if len(X) < 8:
            avg = float(daily.tail(min(7, len(daily))).mean())
            future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
            preds = np.repeat(max(0, avg), horizon)
            method = "Recent-average baseline"
            mae = np.nan
        else:
            split = max(1, int(len(X) * 0.8))
            # Chronological holdout for a realistic error estimate.
            if len(X) - split >= 1:
                eval_model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1)
                eval_model.fit(X.iloc[:split], y[:split])
                eval_pred = np.maximum(0, eval_model.predict(X.iloc[split:]))
                mae = float(mean_absolute_error(y[split:], eval_pred))
                residuals = y[split:] - eval_pred
            else:
                mae = np.nan
                residuals = np.array([])
            model = RandomForestRegressor(n_estimators=220, min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(X, y)
            history = daily.tolist()
            future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
            preds_list = []
            for date in future_dates:
                row = pd.DataFrame([_feature_row(history, date)])
                pred = max(0.0, float(model.predict(row[X.columns])[0]))
                preds_list.append(pred)
                history.append(pred)
            preds = np.array(preds_list)
            method = "Random Forest (lag + calendar features)"
    # Approximate 80% prediction interval from chronological holdout residuals; fallback uses historical variation.
    if 'residuals' in locals() and len(residuals) >= 5:
        lo_err, hi_err = np.quantile(residuals, [0.10, 0.90])
        lower = np.maximum(0, preds + lo_err)
        upper = np.maximum(lower, preds + hi_err)
    else:
        sigma = float(daily.std(ddof=0)) if len(daily) > 1 else float(daily.mean() * 0.25)
        lower = np.maximum(0, preds - 1.28 * sigma)
        upper = np.maximum(lower, preds + 1.28 * sigma)
    output = pd.DataFrame({
        "date": future_dates,
        "product_id": product,
        "forecast": np.round(preds, 2),
        "lower_80": np.round(lower, 2),
        "upper_80": np.round(upper, 2),
    })
    metrics = {"product_id": product, "model": method, "mae": mae, "history_days": len(daily),
               "average_daily_demand": float(daily.mean()), "std_daily_demand": float(daily.std(ddof=0))}
    return output, metrics

def forecast_all(df: pd.DataFrame, horizon: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, metrics = [], []
    for _, group in df.groupby("product_id"):
        result, metric = forecast_product(group, horizon)
        frames.append(result)
        metrics.append(metric)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)
