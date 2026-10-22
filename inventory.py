import numpy as np
import pandas as pd

def build_inventory_recommendations(sales_df: pd.DataFrame, forecast_df: pd.DataFrame,
                                    service_level_z: float = 1.65,
                                    review_period_days: int = 7) -> pd.DataFrame:
    """Compute transparent reorder recommendations from recent demand and lead time."""
    latest = sales_df.sort_values("date").groupby("product_id", as_index=False).tail(1)
    rows = []
    for _, item in latest.iterrows():
        pid = str(item["product_id"])
        hist = sales_df[sales_df["product_id"] == pid].sort_values("date")
        recent = hist.tail(28)["sales"].astype(float)
        avg = float(recent.mean()) if len(recent) else 0.0
        std = float(recent.std(ddof=0)) if len(recent) else 0.0
        lead = int(max(1, item.get("lead_time_days", 7)))
        stock = float(max(0, item.get("inventory", 0)))
        cost = float(max(0, item.get("unit_cost", 0)))
        safety_stock = service_level_z * std * np.sqrt(lead)
        reorder_point = avg * lead + safety_stock
        f = forecast_df[forecast_df["product_id"].astype(str) == pid]
        expected_lead_demand = float(f.head(lead)["forecast"].sum()) if not f.empty else avg * lead
        target_stock = avg * (lead + review_period_days) + safety_stock
        order_qty = max(0, int(np.ceil(target_stock - stock)))
        coverage = stock / avg if avg > 0 else np.inf
        if stock <= reorder_point:
            status = "REORDER NOW"
        elif coverage <= lead + review_period_days:
            status = "WATCH"
        else:
            status = "HEALTHY"
        rows.append({
            "product_id": pid, "current_stock": round(stock, 1),
            "avg_daily_demand_28d": round(avg, 2), "lead_time_days": lead,
            "safety_stock": int(np.ceil(safety_stock)),
            "reorder_point": int(np.ceil(reorder_point)),
            "forecast_demand_during_lead_time": round(expected_lead_demand, 1),
            "days_of_cover": round(coverage, 1) if np.isfinite(coverage) else np.inf,
            "recommended_order_qty": order_qty,
            "estimated_order_value": round(order_qty * cost, 2),
            "status": status,
        })
    return pd.DataFrame(rows).sort_values(["status", "recommended_order_qty"], ascending=[True, False])
