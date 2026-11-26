"""Day-ahead forecasts of site load, PV and the JEPX price, for rolling re-planning.

All forecasters share one contract:

    forecast(history, future_ts) -> DataFrame with columns TARGETS, one row per future timestamp

`history` holds hourly actuals strictly *before* `future_ts[0]`, so a forecaster cannot see the future
by construction (the simulator slices it). Leads are 0..23 h, so "the same hour yesterday" is always known.

  * PublishedPrices     - wrapper: JEPX day-ahead prices are used where already published (10:00 the day before)
  * PerfectForecaster   - returns the actuals (oracle; measures the cost of the rolling horizon alone)
  * PersistenceForecaster - same hour yesterday (the standard naive baseline)
  * GBMForecaster       - gradient-boosted trees (scikit-learn's HistGradientBoostingRegressor, the
                          LightGBM algorithm) on calendar + lag features, one direct model per target
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = ("load_kw", "solar_kw", "tokyo_price_yen_kwh")
MAX_LEAD = 24  # hours


def site_prices(tokyo_price: np.ndarray, wheeling_fee: float = 8.5) -> tuple[np.ndarray, np.ndarray]:
    """Import / export prices from the JEPX Tokyo price (same rule as data/prepare_japan_grid_data.py)."""
    p = np.asarray(tokyo_price, dtype=float)
    return p + wheeling_fee, np.minimum(p * 0.7, 10.0)


class PerfectForecaster:
    name = "perfect forecast"

    def __init__(self, actual: pd.DataFrame):
        self._actual = actual.set_index("timestamp")[list(TARGETS)]

    def forecast(self, history: pd.DataFrame, future_ts: pd.DatetimeIndex) -> pd.DataFrame:
        return self._actual.loc[future_ts].reset_index(drop=True)


class PersistenceForecaster:
    name = "persistence (same hour yesterday)"

    def forecast(self, history: pd.DataFrame, future_ts: pd.DatetimeIndex) -> pd.DataFrame:
        h = history.set_index("timestamp")
        yesterday = pd.DatetimeIndex(np.asarray(future_ts, dtype="datetime64[ns]") - np.timedelta64(24, "h"))
        out = {col: h[col].reindex(yesterday).to_numpy() for col in TARGETS}
        return pd.DataFrame(out).ffill().bfill()


class PublishedPrices:
    """Wrap a forecaster: use JEPX day-ahead prices wherever they are already published.

    The JEPX day-ahead auction closes at 10:00 JST and publishes the next day's 48 half-hour prices shortly
    after. So when deciding the hour that starts at `now`, prices are known up to the end of today, and up to
    the end of tomorrow once it is 10:00 or later. Only the hours beyond that need a price forecast.
    """

    def __init__(self, inner, actual: pd.DataFrame, publish_hour: int = 10):
        self.inner, self.publish_hour = inner, publish_hour
        self.name = inner.name
        self._price = actual.set_index("timestamp")["tokyo_price_yen_kwh"]

    def known_until(self, now: pd.Timestamp) -> pd.Timestamp:
        days = 2 if now.hour >= self.publish_hour else 1
        return pd.Timestamp(np.datetime64(now.normalize(), "ns") + np.timedelta64(days, "D"))

    def forecast(self, history: pd.DataFrame, future_ts: pd.DatetimeIndex) -> pd.DataFrame:
        fc = self.inner.forecast(history, future_ts)
        known = np.asarray(future_ts < self.known_until(future_ts[0]))
        fc.loc[known, "tokyo_price_yen_kwh"] = self._price.reindex(future_ts[known]).to_numpy()
        return fc


# --------------------------------------------------------------------------- gradient boosting

FEATURES = ("lead", "hour", "dow", "day_off", "day_off_lag24", "lag24", "lag48", "lag168", "last", "mean24")

# Japanese public holidays in 2026 (incl. substitute and "sandwiched" days), plus Obon and the New Year break,
# when most businesses close. Load drops on these days, and "same hour yesterday" is wrong the day after.
DAYS_OFF_2026 = pd.to_datetime(
    [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-12",
        "2026-02-11",
        "2026-02-23",
        "2026-03-20",
        "2026-04-29",
        "2026-05-04",
        "2026-05-05",
        "2026-05-06",
        "2026-07-20",
        "2026-08-11",
        "2026-08-13",
        "2026-08-14",
        "2026-08-15",
        "2026-09-21",
        "2026-09-22",
        "2026-09-23",
        "2026-10-12",
        "2026-11-03",
        "2026-11-23",
        "2026-12-29",
        "2026-12-30",
        "2026-12-31",
    ]  # fmt: skip
).date


def day_off(ts: pd.DatetimeIndex) -> np.ndarray:
    """1.0 on weekends, public holidays and the usual company holidays (Obon, New Year)."""
    return ((ts.dayofweek >= 5) | np.isin(ts.date, DAYS_OFF_2026)).astype(float)


def _features(y: np.ndarray, origin: np.ndarray, target: np.ndarray, ts: pd.DatetimeIndex) -> np.ndarray:
    """Feature rows for forecasting y[target] from information up to origin-1 (positions in y)."""

    def at(pos):
        pos = np.asarray(pos)
        ok = pos >= 0
        out = np.full(pos.shape, np.nan)
        out[ok] = y[pos[ok]]
        return out

    # Mean of the last 24 observed values (NaN-aware: load/PV are missing before the site was metered).
    csum = np.concatenate([[0.0], np.nancumsum(y)])
    cnt = np.concatenate([[0], np.cumsum(~np.isnan(y))])
    lo = np.maximum(origin - 24, 0)
    n_obs = cnt[origin] - cnt[lo]
    mean24 = np.where(n_obs > 0, (csum[origin] - csum[lo]) / np.maximum(n_obs, 1), np.nan)
    tts = ts[target]
    return np.column_stack(
        [
            target - origin,
            tts.hour,
            tts.dayofweek,
            day_off(tts),
            day_off(pd.DatetimeIndex(np.asarray(tts, dtype="datetime64[ns]") - np.timedelta64(24, "h"))),
            at(target - 24),
            at(target - 48),
            at(target - 168),
            at(origin - 1),
            mean24,
        ]
    )


@dataclass
class GBMForecaster:
    """One direct multi-lead model per target, trained on every (origin, lead) pair in the history.

    Each model predicts the *correction* to "same hour yesterday" (y - lag24), not y itself. With only
    two weeks of site data, learning the level from scratch is worse than persistence; learning the
    correction starts from persistence and only moves away from it where the features justify it.

    The last `validation_h` hours of the history are held out first: a target whose model does not beat
    persistence there uses persistence instead (PV without a weather forecast is the usual case).
    """

    name: str = "gradient boosting"
    max_iter: int = 300
    learning_rate: float = 0.05
    min_history_h: int = 48
    validation_h: int = 336  # capped at a quarter of the history for short histories
    models: dict = field(default_factory=dict)
    choice: dict = field(default_factory=dict)  # target -> "gbm" | "persistence"
    validation_mae: dict = field(default_factory=dict)  # target -> {"gbm": .., "persistence": ..}

    def _rows(self, y, ts, origins, n_end):
        X, t_all = [], []
        for lead in range(MAX_LEAD):
            o = origins[origins + lead < n_end]
            t = o + lead
            X.append(_features(y, o, t, ts))
            t_all.append(t)
        return np.vstack(X), np.concatenate(t_all)

    def _train(self, y, ts, n_end):
        X, t = self._rows(y, ts, np.arange(max(self.min_history_h, 24), n_end), n_end)
        target = y[t] - y[t - 24]
        keep = ~np.isnan(target)  # e.g. load/PV only exist from the site's first metered day
        model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=0)
        return model.fit(X[keep], target[keep])

    def fit(self, history: pd.DataFrame) -> "GBMForecaster":
        ts = pd.DatetimeIndex(history["timestamp"])
        n = len(history)
        cut = n - min(self.validation_h, n // 4)
        lag = FEATURES.index("lag24")
        for col in TARGETS:
            y = history[col].to_numpy(dtype=float)
            model = self._train(y, ts, cut)
            X, t = self._rows(y, ts, np.arange(cut, n), n)
            ok = ~np.isnan(y[t]) & ~np.isnan(X[:, lag])
            base = X[ok, lag]
            scores = {
                "gbm": float(np.mean(np.abs(y[t][ok] - (base + model.predict(X[ok]))))),
                "persistence": float(np.mean(np.abs(y[t][ok] - base))),
            }
            self.validation_mae[col] = scores
            self.choice[col] = "gbm" if scores["gbm"] < scores["persistence"] else "persistence"
            self.models[col] = self._train(y, ts, n)  # refit on everything
        return self

    def forecast(self, history: pd.DataFrame, future_ts: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("call fit() first")
        ts = pd.DatetimeIndex(list(history["timestamp"]) + list(future_ts))
        n_hist, H = len(history), len(future_ts)
        origin = np.full(H, n_hist)
        target = n_hist + np.arange(H)
        out = {}
        for col in TARGETS:
            y = np.concatenate([history[col].to_numpy(dtype=float), np.full(H, np.nan)])
            X = _features(y, origin, target, ts)
            pred = X[:, FEATURES.index("lag24")]
            if self.choice.get(col, "gbm") == "gbm":
                pred = pred + self.models[col].predict(X)
            pred = pd.Series(pred).ffill().bfill().to_numpy()
            out[col] = np.clip(pred, 0.0, None) if col != "tokyo_price_yen_kwh" else pred
        return pd.DataFrame(out)


def mae(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
    return {c: float(np.mean(np.abs(actual[c].to_numpy() - predicted[c].to_numpy()))) for c in TARGETS}
