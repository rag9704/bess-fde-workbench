"""Rolling-horizon re-planning (model predictive control) of the battery, settled against actual data.

Every hour:
  1. forecast the next `horizon` hours from data strictly before now,
  2. solve the dispatch MILP on the forecast, starting from the *measured* SoC and carrying the
     billing period's peak import so far (raising it costs the demand charge; staying below is free),
  3. apply only the first hour's battery power to the *actual* load, PV and price; the grid absorbs
     the forecast error (import, or export up to the cap, then curtailment),
  4. move on one hour and re-plan.

Peak guard (on by default): the plan is made on forecasts, so the actual import can overshoot the peak the
plan intended, and one overshoot sets the demand charge for the whole month. Real site controllers run a fast
feedback layer under the planner for this. Before settling an hour, if actual import would exceed
max(peak so far, the plan's own peak), the guard first cuts charging, then adds discharge, within the battery's
power and the energy above its reserve floor.

Each plan must end with the site's terminal reserve (`end_soc_min_kwh`), so the battery never drifts
empty at the end of a horizon. The schedule is then scored with the same cost function as the baselines.
"""

from dataclasses import dataclass, field, replace
import time

import numpy as np
import pandas as pd

from core_model.evaluation import CostBreakdown, Schedule, _settle_with_grid, cost_breakdown
from core_model.formulation import BessParams
from core_model.solver_runner import SolverOptions, solve_dispatch

from .forecasters import TARGETS, mae, site_prices

PRICE_COLS = ("import_price_yen_kwh", "export_price_yen_kwh")


@dataclass
class MPCResult:
    name: str
    schedule: Schedule
    cost: CostBreakdown
    solves: int
    solve_sec: float
    fallback_steps: int  # hours where no plan was found and the battery idled
    recovered_steps: int  # hours where the forecast plan needed the elastic (slack) model
    guard_steps: int = 0  # hours where the peak guard changed the planned battery power
    forecast_mae: dict[str, float] = field(default_factory=dict)


def forecast_window(forecaster, history: pd.DataFrame, future_ts: pd.DatetimeIndex, wheeling_fee: float):
    fc = forecaster.forecast(history, future_ts)
    imp, exp = site_prices(fc["tokyo_price_yen_kwh"].to_numpy(), wheeling_fee)
    return pd.DataFrame(
        {
            "timestamp": future_ts,
            "load_kw": fc["load_kw"].to_numpy(),
            "solar_kw": fc["solar_kw"].to_numpy(),
            "tokyo_price_yen_kwh": fc["tokyo_price_yen_kwh"].to_numpy(),
            "import_price_yen_kwh": imp,
            "export_price_yen_kwh": exp,
        }
    )


def run_mpc(
    actual: pd.DataFrame,
    params: BessParams,
    forecaster,
    start: int,
    steps: int,
    horizon: int = 24,
    solver: str = "highs",
    options: SolverOptions | None = None,
    wheeling_fee: float = 8.5,
    history: pd.DataFrame | None = None,
    peak_guard: bool = True,
) -> MPCResult:
    """Simulate `steps` hours of re-planning starting at row `start` of `actual` (hourly data).

    `history` may extend `actual` backwards (e.g. months of JEPX prices before the site was metered);
    it must contain `actual`'s rows too. Forecasters only ever see rows before the current hour.
    """
    if abs(params.dt_hours - 1.0) > 1e-9:
        raise ValueError("rolling re-planning is implemented for hourly data")
    hist = (actual if history is None else history).reset_index(drop=True)
    offset = int(np.searchsorted(hist["timestamp"].to_numpy(), actual["timestamp"].to_numpy()[0]))
    T = steps
    imp, exp, cur = np.zeros(T), np.zeros(T), np.zeros(T)
    ch, dis = np.zeros(T), np.zeros(T)
    soc = np.zeros(T + 1)
    soc[0] = params.init_soc_kwh
    peak = 0.0
    solve_sec, fallback, recovered, guarded = 0.0, 0, 0, 0
    floor = params.default_soc_min_frac * params.capacity_kwh
    fc_rows, act_rows = [], []

    for k in range(T):
        i = start + k
        H = min(horizon, len(actual) - i)
        future_ts = pd.DatetimeIndex(actual["timestamp"].iloc[i : i + H])
        window = forecast_window(forecaster, hist.iloc[: offset + i], future_ts, wheeling_fee)
        fc_rows.append(window[list(TARGETS)])
        act_rows.append(actual[list(TARGETS)].iloc[i : i + H].reset_index(drop=True))

        p = replace(params, init_soc_kwh=float(soc[k]))
        t0 = time.perf_counter()
        strict, recovery = solve_dispatch(window, p, solver=solver, options=options, prior_peak_kw=peak)
        solve_sec += time.perf_counter() - t0
        plan = strict if strict.usable else recovery
        if plan is None or not plan.usable:
            fallback += 1
            c, d, target = 0.0, 0.0, peak
        else:
            recovered += plan is recovery
            c, d = float(plan.p_ch_kw[0]), float(plan.p_dis_kw[0])
            target = max(peak, float(np.max(plan.p_imp_kw)))

        row = actual.iloc[i]
        base_net = float(row["load_kw"]) - float(row["solar_kw"])
        if peak_guard and base_net + c - d > target + 1e-6:
            c, d = guard(base_net, c, d, target, float(soc[k]), floor, params)
            guarded += 1
        net = base_net + c - d
        imp[k], exp[k], cur[k] = _settle_with_grid(net, params)
        ch[k], dis[k] = c, d
        soc[k + 1] = soc[k] + params.charge_eff * c - d / params.discharge_eff
        peak = max(peak, imp[k])

    window_actual = actual.iloc[start : start + T].reset_index(drop=True)
    label = forecaster.name if peak_guard else f"{forecaster.name}, no peak guard"
    schedule = Schedule(f"MPC, {label}", imp, exp, ch, dis, soc, cur)
    cost = cost_breakdown(window_actual, imp, exp, ch, dis, soc, params)
    fc_all, act_all = pd.concat(fc_rows, ignore_index=True), pd.concat(act_rows, ignore_index=True)
    return MPCResult(schedule.name, schedule, cost, T, solve_sec, fallback, recovered, guarded, mae(act_all, fc_all))


def guard(base_net: float, c: float, d: float, target: float, soc: float, floor: float, params: BessParams):
    """Lower charging, then raise discharging, so that import stays at or below `target` if possible."""
    excess = base_net + c - d - target
    cut = min(c, excess)
    c -= cut
    excess -= cut
    if excess > 0 and c <= 1e-9:
        energy = max(soc - floor, 0.0) * params.discharge_eff  # kWh deliverable this hour (dt = 1 h)
        d += max(0.0, min(excess, params.max_discharge_kw - d, energy - d))
    return c, d
