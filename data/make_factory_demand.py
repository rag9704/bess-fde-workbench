"""Synthetic daily demand for the production/inventory example (deterministic, seed 7).

    python data/make_factory_demand.py   # writes data/factory_demand_daily.csv

8 weeks: weekday orders around 100 units, light weekends, a promotion in week 5 and a holiday
week (Golden Week) in week 2 when the line's unit cost rises (overtime / holiday pay).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make(days: int = 56, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-04-20", periods=days, freq="D")
    weekday = dates.dayofweek < 5
    demand = np.where(weekday, 100.0, 30.0) + rng.normal(0, 12, days)
    week = (np.arange(days) // 7) + 1
    demand[week == 5] *= 1.45  # promotion
    demand = np.clip(demand, 0, None).round(0)
    holiday = (dates >= "2026-04-29") & (dates <= "2026-05-06")  # Golden Week
    unit_cost = np.where(holiday | ~weekday, 75.0, 50.0)  # weekend / holiday pay
    supplier_price = np.full(days, 95.0)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "demand_units": demand,
            "line_unit_cost": unit_cost,
            "supplier_price": supplier_price,
        }
    )


if __name__ == "__main__":
    out = Path(__file__).with_name("factory_demand_daily.csv")
    make().to_csv(out, index=False)
    print(f"Wrote {out}")
