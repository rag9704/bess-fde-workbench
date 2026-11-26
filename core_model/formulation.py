"""JijModeling formulation of behind-the-meter BESS + PV dispatch.

One symbolic model serves both the strict and the elastic (slack) variant:
the slack variables' upper bound is a placeholder (`slack_ub_*`). Setting it to
0 gives the strict model; setting it large lets the solver violate the SoC
reserve / grid import cap at a heavy penalty so we can see *where* a scenario
is infeasible.

The model is horizon-agnostic (`T` is a placeholder), so it is built once and
re-used for every site, horizon and scenario: only the instance data changes.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jijmodeling as jm
import numpy as np


@dataclass(frozen=True)
class BessParams:
    """Parameters for a commercial behind-the-meter BESS + PV dispatch horizon."""

    capacity_kwh: float = 2000.0
    max_charge_kw: float = 500.0
    max_discharge_kw: float = 500.0
    charge_eff: float = 0.94
    discharge_eff: float = 0.94
    init_soc_kwh: float = 1000.0
    end_soc_min_kwh: float = 800.0
    grid_import_cap_kw: float = 1200.0
    grid_export_cap_kw: float = 400.0
    peak_demand_charge_yen_kw: float = 1400.0  # Demand charge on the horizon's peak import
    degradation_cost_yen_kwh: float = 2.0  # Throughput wear cost (charge + discharge)
    slack_penalty_yen_kwh: float = 10000.0  # High penalty so slack is only used for recovery
    default_soc_min_frac: float = 0.10  # Default reserve floor as a fraction of capacity
    dt_hours: float = 1.0


@lru_cache(maxsize=1)
def build_problem() -> jm.Problem:
    """Build (once) the symbolic MILP for BESS + grid dispatch."""
    problem = jm.Problem("BESS_Grid_Dispatch", sense=jm.ProblemSense.MINIMIZE)

    @problem.update
    def _(problem: jm.DecoratedProblem):
        # ---- Placeholders (instance data) ----
        T = problem.Natural(description="Number of time steps in the horizon")
        load = problem.Float(shape=(T,), description="Site load [kW]")
        solar = problem.Float(shape=(T,), description="PV generation [kW]")
        price_imp = problem.Float(shape=(T,), description="Import price [yen/kWh]")
        price_exp = problem.Float(shape=(T,), description="Export price [yen/kWh]")
        soc_min = problem.Float(shape=(T + 1,), description="SoC reserve floor per state [kWh]")
        soc_max = problem.Float(shape=(T + 1,), description="SoC ceiling per state [kWh]")
        grid_imp_cap = problem.Float(shape=(T,), description="Grid import cap per step [kW]")

        cap_kwh = problem.Float(description="Battery energy capacity [kWh]")
        p_ch_max = problem.Float(description="Max charge power [kW]")
        p_dis_max = problem.Float(description="Max discharge power [kW]")
        eta_ch = problem.Float(description="Charge efficiency")
        eta_dis = problem.Float(description="Discharge efficiency")
        soc_init = problem.Float(description="Initial SoC [kWh]")
        soc_end_min = problem.Float(description="Minimum terminal SoC [kWh]")
        p_exp_max = problem.Float(description="Grid export cap [kW]")
        p_imp_bound = problem.Float(description="Physical upper bound on import [kW]")
        prior_peak = problem.Float(description="Peak import already set earlier in the billing period [kW]")
        c_peak = problem.Float(description="Demand charge [yen/kW]")
        c_deg = problem.Float(description="Degradation cost [yen/kWh]")
        m_pen = problem.Float(description="Slack penalty [yen/kWh]")
        slack_ub_soc = problem.Float(description="Upper bound of SoC-reserve slack (0 = strict)")
        slack_ub_imp = problem.Float(description="Upper bound of import-cap slack (0 = strict)")
        dt = problem.Float(description="Time step length [h]")

        # ---- Decision variables ----
        p_imp = problem.ContinuousVar(shape=(T,), lower_bound=0, upper_bound=p_imp_bound)
        p_exp = problem.ContinuousVar(shape=(T,), lower_bound=0, upper_bound=p_exp_max)
        p_ch = problem.ContinuousVar(shape=(T,), lower_bound=0, upper_bound=p_ch_max)
        p_dis = problem.ContinuousVar(shape=(T,), lower_bound=0, upper_bound=p_dis_max)
        soc = problem.ContinuousVar(shape=(T + 1,), lower_bound=0, upper_bound=cap_kwh)
        z_ch = problem.BinaryVar(shape=(T,), description="1 = charging mode, 0 = discharging mode")
        # The demand charge is billed on the billing period's peak. In rolling re-planning, the peak already
        # reached is a floor: raising it costs c_peak per kW, staying below it is free.
        p_peak = problem.ContinuousVar(lower_bound=prior_peak, upper_bound=p_imp_bound)
        s_soc_min = problem.ContinuousVar(shape=(T + 1,), lower_bound=0, upper_bound=slack_ub_soc)
        s_imp_cap = problem.ContinuousVar(shape=(T,), lower_bound=0, upper_bound=slack_ub_imp)

        # ---- Objective: energy cost - export revenue + wear + demand charge + slack penalty ----
        problem += (
            jm.sum(
                price_imp[t] * p_imp[t] * dt
                - price_exp[t] * p_exp[t] * dt
                + c_deg * (p_ch[t] + p_dis[t]) * dt
                + m_pen * s_imp_cap[t] * dt
                for t in T
            )
            + m_pen * jm.sum(s_soc_min[k] for k in T + 1)
            + c_peak * p_peak
        )

        # ---- Constraints ----
        problem += problem.Constraint(
            "power_balance",
            [load[t] + p_ch[t] + p_exp[t] == solar[t] + p_dis[t] + p_imp[t] for t in T],
        )
        problem += problem.Constraint("soc_init_eq", soc[0] == soc_init)
        problem += problem.Constraint("soc_end_floor", soc[T] >= soc_end_min)
        problem += problem.Constraint(
            "soc_dynamics",
            [soc[t + 1] == soc[t] + eta_ch * p_ch[t] * dt - (1.0 / eta_dis) * p_dis[t] * dt for t in T],
        )
        # Mode lock: no simultaneous charge & discharge
        problem += problem.Constraint("charge_mode_limit", [p_ch[t] <= p_ch_max * z_ch[t] for t in T])
        problem += problem.Constraint("discharge_mode_limit", [p_dis[t] <= p_dis_max * (1 - z_ch[t]) for t in T])
        problem += problem.Constraint("soc_upper_bound", [soc[k] <= soc_max[k] for k in T + 1])
        problem += problem.Constraint("peak_import_def", [p_peak >= p_imp[t] for t in T])
        # Soft-able operational limits (slack is fixed to 0 in the strict variant)
        problem += problem.Constraint("soc_min_reserve", [soc[k] + s_soc_min[k] >= soc_min[k] for k in T + 1])
        problem += problem.Constraint("grid_import_cap", [p_imp[t] - s_imp_cap[t] <= grid_imp_cap[t] for t in T])

    return problem


def make_instance_data(
    df_slice: Any,
    params: BessParams,
    soc_min_kwh: np.ndarray | None = None,
    soc_max_kwh: np.ndarray | None = None,
    grid_import_cap_kw: np.ndarray | None = None,
    elastic: bool = False,
    prior_peak_kw: float = 0.0,
) -> dict[str, Any]:
    """Convert a time-series DataFrame slice + operational overrides into model inputs.

    `soc_min_kwh` may have length T (per step; the terminal state gets
    max(end_soc_min, last value)) or T+1 (per state).
    `soc_max_kwh` (optional) follows the same length rules and caps the SoC per state.
    `prior_peak_kw` is the import peak already reached in the billing period (0 for a fresh period).
    """
    T = len(df_slice)
    s_min = np.full(T + 1, params.default_soc_min_frac * params.capacity_kwh, dtype=float)
    if soc_min_kwh is not None:
        arr = np.asarray(soc_min_kwh, dtype=float)
        if len(arr) == T + 1:
            s_min = arr.copy()
        elif len(arr) == T:
            s_min[:T] = arr
            s_min[T] = max(params.end_soc_min_kwh, arr[-1])
        else:
            raise ValueError(f"soc_min_kwh must have length {T} or {T + 1}, got {len(arr)}")

    cap = float(params.capacity_kwh)
    s_max = np.full(T + 1, cap, dtype=float)
    if soc_max_kwh is not None:
        arr = np.asarray(soc_max_kwh, dtype=float)
        if len(arr) == T + 1:
            s_max = np.minimum(arr, cap)
        elif len(arr) == T:
            s_max[:T] = np.minimum(arr, cap)
        else:
            raise ValueError(f"soc_max_kwh must have length {T} or {T + 1}, got {len(arr)}")

    g_cap = np.full(T, params.grid_import_cap_kw, dtype=float)
    if grid_import_cap_kw is not None:
        g_cap = np.asarray(grid_import_cap_kw, dtype=float)
        if len(g_cap) != T:
            raise ValueError(f"grid_import_cap_kw must have length {T}, got {len(g_cap)}")

    load = df_slice["load_kw"].to_numpy(dtype=float)
    # Physically the site can never import more than load + full charge + full export.
    p_imp_bound = float(max(np.max(load) + params.max_charge_kw + params.grid_export_cap_kw, prior_peak_kw))

    return {
        "T": T,
        "load": load,
        "solar": df_slice["solar_kw"].to_numpy(dtype=float),
        "price_imp": df_slice["import_price_yen_kwh"].to_numpy(dtype=float),
        "price_exp": df_slice["export_price_yen_kwh"].to_numpy(dtype=float),
        "soc_min": s_min,
        "soc_max": s_max,
        "grid_imp_cap": g_cap,
        "cap_kwh": cap,
        "p_ch_max": float(params.max_charge_kw),
        "p_dis_max": float(params.max_discharge_kw),
        "eta_ch": float(params.charge_eff),
        "eta_dis": float(params.discharge_eff),
        "soc_init": float(params.init_soc_kwh),
        "soc_end_min": float(params.end_soc_min_kwh),
        "p_exp_max": float(params.grid_export_cap_kw),
        "p_imp_bound": p_imp_bound,
        "prior_peak": float(prior_peak_kw),
        "c_peak": float(params.peak_demand_charge_yen_kw),
        "c_deg": float(params.degradation_cost_yen_kwh),
        "m_pen": float(params.slack_penalty_yen_kwh),
        "slack_ub_soc": cap if elastic else 0.0,
        "slack_ub_imp": p_imp_bound if elastic else 0.0,
        "dt": float(params.dt_hours),
    }
