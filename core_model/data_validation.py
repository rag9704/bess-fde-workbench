"""Input checks for site time-series data, run before anything reaches the solver.

Bad customer data is the most common way an optimizer silently produces nonsense,
so we fail early with a list of *all* problems found, in plain language.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "load_kw",
    "solar_kw",
    "import_price_yen_kwh",
    "export_price_yen_kwh",
)


class DataValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Input data failed validation:\n  - " + "\n  - ".join(issues))


def validate_timeseries(df: pd.DataFrame, expected_step_hours: float = 1.0) -> list[str]:
    """Validate a site time series. Raises DataValidationError; returns non-fatal warnings."""
    issues: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError([f"Missing required column(s): {', '.join(missing)}"])
    if df.empty:
        raise DataValidationError(["No rows in the selected window"])

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        issues.append(f"{int(ts.isna().sum())} timestamp(s) could not be parsed")
    else:
        if ts.duplicated().any():
            issues.append(f"{int(ts.duplicated().sum())} duplicated timestamp(s), e.g. {ts[ts.duplicated()].iloc[0]}")
        if not ts.is_monotonic_increasing:
            issues.append("Timestamps are not sorted in increasing order")
        steps = ts.diff().dropna().dt.total_seconds() / 3600.0
        bad = steps[~np.isclose(steps, expected_step_hours)]
        if len(bad):
            first = ts.loc[bad.index[0]]
            issues.append(f"{len(bad)} gap(s)/irregular step(s) (expected {expected_step_hours} h), first at {first}")

    for col in REQUIRED_COLUMNS[1:]:
        vals = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((~np.isfinite(vals)).sum())
        if n_bad:
            issues.append(f"Column '{col}' has {n_bad} missing / non-numeric value(s)")

    for col in ("load_kw", "solar_kw"):
        vals = pd.to_numeric(df[col], errors="coerce")
        if (vals < 0).any():
            issues.append(f"Column '{col}' has {int((vals < 0).sum())} negative value(s)")

    price = pd.to_numeric(df["import_price_yen_kwh"], errors="coerce")
    if (price < 0).any():
        warnings.append(
            f"{int((price < 0).sum())} hour(s) with negative import price; the model handles this, "
            "but check it is intended"
        )
    if (pd.to_numeric(df["export_price_yen_kwh"], errors="coerce") > price).any():
        warnings.append("Some hours have export price > import price (arbitrage via the grid is possible)")

    if issues:
        raise DataValidationError(issues)
    return warnings
