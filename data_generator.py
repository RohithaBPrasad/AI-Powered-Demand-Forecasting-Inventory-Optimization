from pathlib import Path
import numpy as np
import pandas as pd

def make_sample_data(days: int = 540, products: int = 8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=days, freq="D")
    rows = []
    for p in range(1, products + 1):
        pid = f"SKU-{p:03d}"
        base = rng.uniform(12, 65)
        trend = rng.uniform(-0.015, 0.04)
        weekday_effect = np.array([1.0, 1.03, 1.06, 1.08, 1.15, 1.28, 0.82])
        inventory = int(rng.integers(80, 260))
        unit_cost = round(float(rng.uniform(3, 80)), 2)
        lead = int(rng.integers(3, 15))
        for i, date in enumerate(dates):
            yearly = 1 + 0.18 * np.sin(2 * np.pi * i / 365.25)
            annual_trend = 1 + trend * (i / 30)
            promo = int(rng.random() < 0.07)
            promo_effect = 1.5 if promo else 1.0
            mean = max(1, base * yearly * annual_trend * weekday_effect[date.dayofweek] * promo_effect)
            sales = max(0, int(rng.poisson(mean)))
            # occasional artificial supply constraints reflected in on-hand stock
            inventory = max(0, inventory - sales + (int(rng.integers(20, 75)) if inventory < mean * 5 else 0))
            rows.append({"date": date, "product_id": pid, "sales": sales,
                         "inventory": inventory, "unit_cost": unit_cost,
                         "lead_time_days": lead, "promotion": promo})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = Path("data/sample_sales.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample_data().to_csv(out, index=False)
    print(f"Wrote {out}")
