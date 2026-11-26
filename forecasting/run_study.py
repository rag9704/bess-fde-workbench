"""What does uncertainty cost? Rolling re-planning with forecasts vs perfect hindsight.

    python -m forecasting.run_study                  # October 2026, HiGHS via JijModeling (~15 min)
    python -m forecasting.run_study --solver scipy   # same, SciPy/HiGHS directly (~2 min)
    python -m forecasting.run_study --test-start 2026-08-01 --days 7

Train: everything before the test period (site load / PV since April, JEPX prices since January).
Test: `--days` days from `--test-start`, re-planned every hour. The test period must lie inside one
calendar month, because the demand charge is billed on each month's peak.

Policies compared on the same actual data:
  no battery, rule-based, MPC + persistence forecast, MPC + gradient-boosting forecast,
  MPC + perfect forecast (rolling horizon, no forecast error), and the hindsight optimum
  (one solve over the whole test period with the actual data: no rolling horizon, no forecast error).

Writes forecasting/RESULTS.md and forecasting/results/*.csv.
"""

import argparse
import datetime as dt
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from core_model.evaluation import Schedule, cost_breakdown, no_battery_schedule, rule_based_schedule
from core_model.formulation import BessParams
from core_model.solver_runner import SOLVERS, solve_dispatch

from .forecasters import TARGETS, GBMForecaster, PerfectForecaster, PersistenceForecaster, PublishedPrices
from .mpc import run_mpc

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def load_jepx_history() -> pd.DataFrame:
    """Hourly JEPX Tokyo prices for 2026 from the raw file (re-uses the data-prep parser)."""
    spec = importlib.util.spec_from_file_location("prep", ROOT / "data" / "prepare_japan_grid_data.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return prep.load_jepx_hourly(ROOT / "data" / "raw" / "spot_summary_2026.csv")


def build_frames(site_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(actual site data, history = JEPX since January + site load/PV where metered)."""
    actual = pd.read_csv(site_csv, parse_dates=["timestamp"])
    jepx = load_jepx_history()
    jepx = jepx[jepx["timestamp"] <= actual["timestamp"].max()]
    history = jepx.merge(actual[["timestamp", "load_kw", "solar_kw"]], on="timestamp", how="left")
    # Inside the site period use the site file's prices so history and actuals agree exactly.
    history = history.set_index("timestamp")
    history.loc[actual["timestamp"], "tokyo_price_yen_kwh"] = actual.set_index("timestamp")["tokyo_price_yen_kwh"]
    return actual, history.reset_index()[["timestamp", *TARGETS]]


def schedule_row(name, sched: Schedule, window, params, base_cost, best_saving, extra=None) -> dict:
    c = cost_breakdown(window, sched.p_imp_kw, sched.p_exp_kw, sched.p_ch_kw, sched.p_dis_kw, sched.soc_kwh, params)
    saving = base_cost - c.adjusted_cost_yen
    row = {
        "policy": name,
        "cost_yen": round(c.adjusted_cost_yen),
        "saving_vs_no_battery_yen": round(saving),
        "share_of_hindsight_saving_pct": round(100 * saving / best_saving, 1) if best_saving else None,
        "peak_import_kw": round(c.peak_import_kw, 1),
        "end_soc_kwh": round(float(sched.soc_kwh[-1])),
    }
    row.update(extra or {})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "japan_2026_hourly.csv")
    ap.add_argument("--test-start", default="2026-10-01", help="First test day (start of a billing month is best)")
    ap.add_argument("--days", type=int, default=31, help="Test days (hourly re-planning)")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--solver", choices=SOLVERS, default="highs")
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write RESULTS.md")
    args = ap.parse_args(argv)

    params = BessParams()
    actual, history = build_frames(args.data)
    test_start = pd.Timestamp(args.test_start)
    start, steps = int(np.searchsorted(actual["timestamp"].to_numpy(), test_start.to_datetime64())), args.days * 24
    last = pd.Timestamp(np.datetime64(test_start, "ns") + np.timedelta64(steps - 1, "h"))
    if (last.year, last.month) != (test_start.year, test_start.month):
        print("The test period must lie inside one calendar month (one demand-charge billing period).", file=sys.stderr)
        return 2
    if start == 0 or start + steps + args.horizon - 1 > len(actual):
        print(f"Need training data before {test_start:%Y-%m-%d} and {args.horizon} h after the test", file=sys.stderr)
        return 2
    window = actual.iloc[start : start + steps].reset_index(drop=True)
    print(f"Test period {window['timestamp'].iloc[0]} -> {window['timestamp'].iloc[-1]} ({steps} hourly re-plans)")

    t0 = time.perf_counter()
    train = history[history["timestamp"] < window["timestamp"].iloc[0]]
    gbm = GBMForecaster().fit(train)
    fit_sec = time.perf_counter() - t0
    print(f"Trained gradient boosting on {len(train):,} hours in {fit_sec:.1f}s; per target: {gbm.choice}")

    hindsight, _ = solve_dispatch(window, params, solver=args.solver)
    hs_sched = Schedule(
        "hindsight", hindsight.p_imp_kw, hindsight.p_exp_kw, hindsight.p_ch_kw, hindsight.p_dis_kw, hindsight.soc_kwh
    )
    base = no_battery_schedule(window, params)
    base_cost = cost_breakdown(
        window, base.p_imp_kw, base.p_exp_kw, base.p_ch_kw, base.p_dis_kw, base.soc_kwh, params
    ).adjusted_cost_yen
    hs_cost = cost_breakdown(
        window, hs_sched.p_imp_kw, hs_sched.p_exp_kw, hs_sched.p_ch_kw, hs_sched.p_dis_kw, hs_sched.soc_kwh, params
    ).adjusted_cost_yen
    best = base_cost - hs_cost

    rows = [
        schedule_row("No battery", base, window, params, base_cost, best),
        schedule_row(
            "Rule-based (charge cheap hours, discharge expensive)",
            rule_based_schedule(window, params),
            window,
            params,
            base_cost,
            best,
        ),
    ]
    persistence = PublishedPrices(PersistenceForecaster(), actual)
    gbm_published = PublishedPrices(gbm, actual)
    runs = [  # (label, forecaster, peak guard)
        ("MPC + persistence", persistence, True),
        ("MPC + gradient boosting", gbm_published, True),
        ("MPC + perfect forecast", PerfectForecaster(actual), True),
        ("MPC + gradient boosting, JEPX prices forecast too", gbm, True),  # ablation: ignore published prices
        ("MPC + gradient boosting, no peak guard", gbm_published, False),  # ablation: no real-time guard
    ]
    mpc = {}
    for label, f, guard_on in runs:
        t = time.perf_counter()
        r = run_mpc(actual, params, f, start, steps, args.horizon, args.solver, history=history, peak_guard=guard_on)
        print(f"  {label:<52} {time.perf_counter() - t:6.1f}s  cost {r.cost.adjusted_cost_yen:,.0f}")
        mpc[label] = r
        rows.append(
            schedule_row(
                label,
                r.schedule,
                window,
                params,
                base_cost,
                best,
                {
                    "replans": r.solves,
                    "solve_sec": round(r.solve_sec, 1),
                    "recovered": r.recovered_steps,
                    "idle_fallbacks": r.fallback_steps,
                    "guarded_hours": r.guard_steps,
                },
            )
        )
    rows.append(
        schedule_row(
            "Hindsight optimum (one solve, actual data)",
            hs_sched,
            window,
            params,
            base_cost,
            best,
            {"replans": 1, "solve_sec": round(hindsight.solve_time_sec, 1)},
        )
    )
    policies = pd.DataFrame(rows)

    print("Scoring forecasts (no solving)...")
    acc, published_share = forecast_accuracy(
        [("persistence (same hour yesterday)", PersistenceForecaster()), ("gradient boosting", gbm)],
        history,
        actual,
        start,
        steps,
        args.horizon,
        persistence,
    )
    print()
    print(policies.to_string(index=False))
    print()
    print(acc.to_string(index=False))

    if args.no_write:
        return 0
    (OUT / "results").mkdir(exist_ok=True)
    policies.to_csv(OUT / "results" / "policies.csv", index=False)
    acc.to_csv(OUT / "results" / "forecast_accuracy.csv", index=False)
    cost = {r["policy"]: r["cost_yen"] for r in rows}
    gbm_cost = cost["MPC + gradient boosting"]
    gap_forecast = gbm_cost - cost["MPC + perfect forecast"]
    gap_horizon = cost["MPC + perfect forecast"] - cost["Hindsight optimum (one solve, actual data)"]
    gap_guard = cost["MPC + gradient boosting, no peak guard"] - gbm_cost
    gap_prices = cost["MPC + gradient boosting, JEPX prices forecast too"] - gbm_cost
    choices = "\n".join(
        f"- `{k}`: {gbm.choice[k]} (gbm {v['gbm']:.2f} vs persistence {v['persistence']:.2f})"
        for k, v in gbm.validation_mae.items()
    )
    report = f"""# Forecast-driven re-planning: results

Run {dt.datetime.now():%Y-%m-%d %H:%M}, solver `{args.solver}`. Test period {window["timestamp"].iloc[0]:%Y-%m-%d %H:%M}
to {window["timestamp"].iloc[-1]:%Y-%m-%d %H:%M} ({steps} hourly re-plans per MPC policy, {args.horizon} h horizon).
Generated by `python -m forecasting.run_study`.

Costs are over the test period on **actual** data, adjusted for the energy left in the battery at the end
(valued like the baselines in `core_model/evaluation.py`). `share_of_hindsight_saving_pct` is how much of the
best possible saving (hindsight optimum vs no battery) each policy captures.

## Policies

{to_md(policies)}

## Where the gap comes from

- Forecast error (gradient-boosting MPC vs perfect-forecast MPC): **¥{gap_forecast:,.0f}**
- Rolling 24 h horizon and terminal reserve (perfect-forecast MPC vs hindsight): **¥{gap_horizon:,.0f}**
- Saved by the real-time peak guard (gradient-boosting MPC without vs with the guard): **¥{gap_guard:,.0f}**
- Saved by using the published JEPX day-ahead prices instead of forecasting them: **¥{gap_prices:,.0f}**

Gradient boosting per target (chosen on the last {gbm.validation_h} h of training, validation MAE):
{choices}

## Forecast accuracy (all leads 0-23 h, every hour of the test period)

The raw forecasters, before published prices are substituted. Skill = 1 - MAE / MAE(persistence).
Load and PV are scaled TEPCO-area curves (see data/), the price is JEPX Tokyo in yen/kWh.
With the auction clock, **{published_share:.0%}** of the price-hours in the 24 h plans are already published,
so the price forecast only matters for the rest (tomorrow's early hours, before 10:00).

{to_md(acc)}
"""
    (OUT / "RESULTS.md").write_text(report, encoding="utf-8")
    print(f"\nWrote {OUT / 'RESULTS.md'}")
    return 0


def forecast_accuracy(named, history, actual, start, steps, horizon, published) -> tuple[pd.DataFrame, float]:
    """MAE of each raw forecaster over every hourly origin of the test period, and the published price share."""
    offset = int(np.searchsorted(history["timestamp"].to_numpy(), actual["timestamp"].to_numpy()[0]))
    errors = {name: {c: [] for c in TARGETS} for name, _ in named}
    known = total = 0
    for k in range(steps):
        i = start + k
        future_ts = pd.DatetimeIndex(actual["timestamp"].iloc[i : i + horizon])
        truth = actual[list(TARGETS)].iloc[i : i + horizon].reset_index(drop=True)
        known += int((future_ts < published.known_until(future_ts[0])).sum())
        total += len(future_ts)
        for name, f in named:
            fc = f.forecast(history.iloc[: offset + i], future_ts)
            for c in TARGETS:
                errors[name][c].append(np.abs(fc[c].to_numpy() - truth[c].to_numpy()))
    mae = {name: {c: float(np.concatenate(v).mean()) for c, v in e.items()} for name, e in errors.items()}
    base = mae[named[0][0]]
    rows = [
        {
            "forecaster": name,
            **{f"MAE {c}": round(m[c], 2) for c in TARGETS},
            **{f"skill {c} %": round(100 * (1 - m[c] / base[c]), 1) for c in TARGETS},
        }
        for name, m in mae.items()
    ]
    return pd.DataFrame(rows), known / total


def to_md(df: pd.DataFrame) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (int, np.integer)) or (isinstance(v, float) and abs(v) >= 1000):
            return f"{v:,.0f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
