from dataclasses import dataclass
from typing import Any
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
    peak_demand_charge_yen_kw: float = 1400.0  # Monthly/period peak demand weight
    degradation_cost_yen_kwh: float = 2.0    # Throughput wear cost (charge + discharge)
    slack_penalty_yen_kwh: float = 10000.0   # High penalty so slack is only used for recovery
    dt_hours: float = 1.0


def build_jijmodeling_problem(T: int, elastic: bool = False) -> Any:
    """Build the JijModeling MIP formulation for BESS + grid dispatch.

    Raises ImportError if `jijmodeling` is not installed.
    """
    import jijmodeling as jm

    problem = jm.Problem("BESS_Grid_Dispatch_Elastic" if elastic else "BESS_Grid_Dispatch")

    # 1D time-indexed input arrays (length T)
    load = problem.Float("load", shape=(T,))
    solar = problem.Float("solar", shape=(T,))
    price_imp = problem.Float("price_imp", shape=(T,))
    price_exp = problem.Float("price_exp", shape=(T,))
    soc_min = problem.Float("soc_min", shape=(T,))
    soc_max = problem.Float("soc_max", shape=(T,))
    grid_imp_cap = problem.Float("grid_imp_cap", shape=(T,))

    # Scalar parameters
    cap_kwh = problem.Float("cap_kwh")
    p_ch_max = problem.Float("p_ch_max")
    p_dis_max = problem.Float("p_dis_max")
    eta_ch = problem.Float("eta_ch")
    eta_dis = problem.Float("eta_dis")
    soc_init = problem.Float("soc_init")
    soc_end_min = problem.Float("soc_end_min")
    p_exp_max = problem.Float("p_exp_max")
    c_peak = problem.Float("c_peak")
    c_deg = problem.Float("c_deg")
    dt = problem.Float("dt")

    # Decision variables
    p_imp = problem.ContinuousVar("p_imp", shape=(T,), lower_bound=0.0)
    p_exp = problem.ContinuousVar("p_exp", shape=(T,), lower_bound=0.0)
    p_ch = problem.ContinuousVar("p_ch", shape=(T,), lower_bound=0.0)
    p_dis = problem.ContinuousVar("p_dis", shape=(T,), lower_bound=0.0)
    soc = problem.ContinuousVar("soc", shape=(T + 1,), lower_bound=0.0)
    z_ch = problem.BinaryVar("z_ch", shape=(T,))
    p_peak = problem.ContinuousVar("p_peak", lower_bound=0.0)

    t = jm.Element("t", belong_to=(0, T))

    # Objective: net energy cost + battery throughput wear + peak import charge
    energy_and_wear = jm.sum(
        t,
        price_imp[t] * p_imp[t] * dt
        - price_exp[t] * p_exp[t] * dt
        + c_deg * (p_ch[t] + p_dis[t]) * dt,
    )
    objective = energy_and_wear + c_peak * p_peak

    if elastic:
        m_pen = problem.Float("m_pen")
        s_soc_min = problem.ContinuousVar("s_soc_min", shape=(T + 1,), lower_bound=0.0)
        s_imp_cap = problem.ContinuousVar("s_imp_cap", shape=(T,), lower_bound=0.0)
        objective = objective + m_pen * (jm.sum(t, s_imp_cap[t] * dt) + jm.sum(t, s_soc_min[t]))

    problem += objective

    # Power balance at each hour t: load + charge + export == solar + discharge + import
    problem += jm.Constraint(
        "power_balance",
        load[t] + p_ch[t] + p_exp[t] == solar[t] + p_dis[t] + p_imp[t],
        forall=t,
    )

    # Initial & terminal SoC
    problem += jm.Constraint("soc_init_eq", soc[0] == soc_init)
    problem += jm.Constraint("soc_end_floor", soc[T] >= soc_end_min)

    # SoC transition dynamics
    problem += jm.Constraint(
        "soc_dynamics",
        soc[t + 1] == soc[t] + eta_ch * p_ch[t] * dt - (1.0 / eta_dis) * p_dis[t] * dt,
        forall=t,
    )

    # Mode lock (prevents simultaneous charging & discharging)
    problem += jm.Constraint("charge_mode_limit", p_ch[t] <= p_ch_max * z_ch[t], forall=t)
    problem += jm.Constraint("discharge_mode_limit", p_dis[t] <= p_dis_max * (1 - z_ch[t]), forall=t)

    # Peak import definition
    problem += jm.Constraint("peak_import_def", p_peak >= p_imp[t], forall=t)

    # Export cap & battery capacity upper bound
    problem += jm.Constraint("grid_export_cap", p_exp[t] <= p_exp_max, forall=t)
    problem += jm.Constraint("soc_upper_cap", soc[t] <= cap_kwh, forall=t)
    problem += jm.Constraint("soc_upper_bound_param", soc[t] <= soc_max[t], forall=t)

    # SoC reserve floor & grid import cap (strict vs. elastic slack)
    if elastic:
        problem += jm.Constraint("soc_min_reserve", soc[t] >= soc_min[t] - s_soc_min[t], forall=t)
        problem += jm.Constraint("grid_import_cap", p_imp[t] <= grid_imp_cap[t] + s_imp_cap[t], forall=t)
    else:
        problem += jm.Constraint("soc_min_reserve", soc[t] >= soc_min[t], forall=t)
        problem += jm.Constraint("grid_import_cap", p_imp[t] <= grid_imp_cap[t], forall=t)

    return problem


def make_instance_data(
    df_slice: Any,
    params: BessParams,
    soc_min_kwh: np.ndarray | None = None,
    soc_max_kwh: np.ndarray | None = None,
    grid_import_cap_kw: np.ndarray | None = None,
    elastic: bool = False,
) -> dict[str, Any]:
    """Convert a time-series DataFrame slice + operational overrides into model inputs."""
    T = len(df_slice)
    s_min = np.full(T + 1, 0.10 * params.capacity_kwh, dtype=float)
    if soc_min_kwh is not None:
        s_min[:T] = np.asarray(soc_min_kwh, dtype=float)
        s_min[T] = max(params.end_soc_min_kwh, s_min[T - 1])

    s_max = np.full(T + 1, params.capacity_kwh, dtype=float)
    if soc_max_kwh is not None:
        s_max[:T] = np.asarray(soc_max_kwh, dtype=float)

    g_cap = np.full(T, params.grid_import_cap_kw, dtype=float)
    if grid_import_cap_kw is not None:
        g_cap = np.asarray(grid_import_cap_kw, dtype=float)

    data: dict[str, Any] = {
        "load": df_slice["load_kw"].to_numpy(dtype=float),
        "solar": df_slice["solar_kw"].to_numpy(dtype=float),
        "price_imp": df_slice["import_price_yen_kwh"].to_numpy(dtype=float),
        "price_exp": df_slice["export_price_yen_kwh"].to_numpy(dtype=float),
        "soc_min": s_min,
        "soc_max": s_max,
        "grid_imp_cap": g_cap,
        "cap_kwh": float(params.capacity_kwh),
        "p_ch_max": float(params.max_charge_kw),
        "p_dis_max": float(params.max_discharge_kw),
        "eta_ch": float(params.charge_eff),
        "eta_dis": float(params.discharge_eff),
        "soc_init": float(params.init_soc_kwh),
        "soc_end_min": float(params.end_soc_min_kwh),
        "p_exp_max": float(params.grid_export_cap_kw),
        "c_peak": float(params.peak_demand_charge_yen_kw),
        "c_deg": float(params.degradation_cost_yen_kwh),
        "dt": float(params.dt_hours),
    }
    if elastic:
        data["m_pen"] = float(params.slack_penalty_yen_kwh)
    return data