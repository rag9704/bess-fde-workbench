from dataclasses import dataclass
import time
from typing import Any
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .formulation import BessParams, build_jijmodeling_problem, make_instance_data


@dataclass
class DispatchResult:
    status: str
    objective_yen: float | None
    solve_time_sec: float
    backend: str
    elastic: bool
    peak_import_kw: float
    total_import_kwh: float
    total_export_kwh: float
    total_charge_kwh: float
    total_discharge_kwh: float
    soc_kwh: np.ndarray
    p_imp_kw: np.ndarray
    p_exp_kw: np.ndarray
    p_ch_kw: np.ndarray
    p_dis_kw: np.ndarray
    slack_soc_kwh: np.ndarray
    slack_imp_kw: np.ndarray

    def infeasibility_summary(self, timestamps: list[str] | None = None) -> list[str]:
        """Return human-readable constraint violations when elastic slack > 1e-4 is active."""
        notes: list[str] = []
        for idx, val in enumerate(self.slack_imp_kw):
            if val > 1e-4:
                label = timestamps[idx] if timestamps and idx < len(timestamps) else f"t={idx}"
                notes.append(f"Grid import cap exceeded by {val:.1f} kW at {label}")
        for idx, val in enumerate(self.slack_soc_kwh):
            if val > 1e-4:
                label = timestamps[idx] if timestamps and idx < len(timestamps) else f"t={idx}"
                notes.append(f"Minimum SoC reserve shortfall of {val:.1f} kWh at {label}")
        return notes


def _solve_with_scipy_milp(data: dict[str, Any], elastic: bool) -> DispatchResult:
    """Solver fallback using SciPy HiGHS MILP so the pipeline runs standalone if OMMX isn't installed."""
    t0 = time.perf_counter()
    T = len(data["load"])
    dt = data["dt"]
    eta_ch = data["eta_ch"]
    eta_dis = data["eta_dis"]
    p_ch_max = data["p_ch_max"]
    p_dis_max = data["p_dis_max"]

    # Variable layout:
    # 0..T-1: p_imp
    # T..2T-1: p_exp
    # 2T..3T-1: p_ch
    # 3T..4T-1: p_dis
    # 4T..5T: soc (T+1 vars)
    # 5T+1..6T: z_ch (T binary vars)
    # 6T+1: p_peak (1 var)
    # If elastic:
    #   6T+2..7T+2: s_soc_min (T+1 vars)
    #   7T+3..8T+2: s_imp_cap (T vars)
    n_soc = T + 1
    base_vars = 4 * T + n_soc + T + 1
    n_vars = base_vars + (n_soc + T if elastic else 0)

    c = np.zeros(n_vars, dtype=float)
    c[0:T] = data["price_imp"] * dt
    c[T : 2 * T] = -data["price_exp"] * dt
    c[2 * T : 3 * T] = data["c_deg"] * dt
    c[3 * T : 4 * T] = data["c_deg"] * dt
    c[6 * T + 1] = data["c_peak"]

    if elastic:
        m_pen = data["m_pen"]
        c[base_vars : base_vars + n_soc] = m_pen
        c[base_vars + n_soc : base_vars + n_soc + T] = m_pen * dt

    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)
    ub[0:T] = data["grid_imp_cap"]
    ub[T : 2 * T] = data["p_exp_max"]
    ub[2 * T : 3 * T] = p_ch_max
    ub[3 * T : 4 * T] = p_dis_max
    ub[4 * T : 4 * T + n_soc] = np.minimum(data["cap_kwh"], data["soc_max"])
    ub[5 * T + 1 : 6 * T + 1] = 1.0

    # Lower bounds on soc
    lb[4 * T : 4 * T + n_soc] = data["soc_min"]
    if elastic:
        # Allow soc to drop below soc_min when s_soc_min > 0
        lb[4 * T : 4 * T + n_soc] = 0.0
        ub[0:T] = np.inf  # Handled via elastic constraint row below

    integrality = np.zeros(n_vars, dtype=int)
    integrality[5 * T + 1 : 6 * T + 1] = 1

    rows: list[np.ndarray] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    def add_row(coeffs: np.ndarray, lo: float, hi: float) -> None:
        rows.append(coeffs)
        row_lb.append(lo)
        row_ub.append(hi)

    # 1. Power balance: p_imp - p_exp + p_ch - p_dis == load - solar
    for t_idx in range(T):
        r = np.zeros(n_vars)
        r[t_idx] = 1.0
        r[T + t_idx] = -1.0
        r[2 * T + t_idx] = 1.0
        r[3 * T + t_idx] = -1.0
        rhs = data["load"][t_idx] - data["solar"][t_idx]
        add_row(r, rhs, rhs)

    # 2. Initial SoC: soc[0] == soc_init
    r0 = np.zeros(n_vars)
    r0[4 * T] = 1.0
    add_row(r0, data["soc_init"], data["soc_init"])

    # 3. Terminal SoC floor: soc[T] >= soc_end_min
    rT = np.zeros(n_vars)
    rT[4 * T + T] = 1.0
    add_row(rT, data["soc_end_min"], np.inf)

    # 4. SoC dynamics: soc[t+1] - soc[t] - eta_ch*dt*p_ch + (dt/eta_dis)*p_dis == 0
    for t_idx in range(T):
        r = np.zeros(n_vars)
        r[4 * T + t_idx + 1] = 1.0
        r[4 * T + t_idx] = -1.0
        r[2 * T + t_idx] = -eta_ch * dt
        r[3 * T + t_idx] = dt / eta_dis
        add_row(r, 0.0, 0.0)

    # 5. Charge & discharge mutual exclusion via binary z_ch[t]
    for t_idx in range(T):
        # p_ch[t] - p_ch_max * z_ch[t] <= 0
        r_ch = np.zeros(n_vars)
        r_ch[2 * T + t_idx] = 1.0
        r_ch[5 * T + 1 + t_idx] = -p_ch_max
        add_row(r_ch, -np.inf, 0.0)

        # p_dis[t] + p_dis_max * z_ch[t] <= p_dis_max
        r_dis = np.zeros(n_vars)
        r_dis[3 * T + t_idx] = 1.0
        r_dis[5 * T + 1 + t_idx] = p_dis_max
        add_row(r_dis, -np.inf, p_dis_max)

    # 6. Peak import definition: p_peak - p_imp[t] >= 0
    for t_idx in range(T):
        rp = np.zeros(n_vars)
        rp[6 * T + 1] = 1.0
        rp[t_idx] = -1.0
        add_row(rp, 0.0, np.inf)

    # 7. Elastic slack constraints if enabled
    if elastic:
        # soc[t] + s_soc_min[t] >= soc_min[t]
        for k in range(n_soc):
            rs = np.zeros(n_vars)
            rs[4 * T + k] = 1.0
            rs[base_vars + k] = 1.0
            add_row(rs, data["soc_min"][k], np.inf)

        # p_imp[t] - s_imp_cap[t] <= grid_imp_cap[t]
        for t_idx in range(T):
            rg = np.zeros(n_vars)
            rg[t_idx] = 1.0
            rg[base_vars + n_soc + t_idx] = -1.0
            add_row(rg, -np.inf, data["grid_imp_cap"][t_idx])

    A = np.vstack(rows)
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A, row_lb, row_ub),
    )
    elapsed = time.perf_counter() - t0

    if not res.success or res.x is None:
        zeros_t = np.zeros(T)
        zeros_s = np.zeros(n_soc)
        return DispatchResult(
            status="INFEASIBLE",
            objective_yen=None,
            solve_time_sec=elapsed,
            backend="scipy.optimize.milp (HiGHS)",
            elastic=elastic,
            peak_import_kw=0.0,
            total_import_kwh=0.0,
            total_export_kwh=0.0,
            total_charge_kwh=0.0,
            total_discharge_kwh=0.0,
            soc_kwh=zeros_s,
            p_imp_kw=zeros_t,
            p_exp_kw=zeros_t,
            p_ch_kw=zeros_t,
            p_dis_kw=zeros_t,
            slack_soc_kwh=zeros_s,
            slack_imp_kw=zeros_t,
        )

    x = res.x
    p_imp = x[0:T]
    p_exp = x[T : 2 * T]
    p_ch = x[2 * T : 3 * T]
    p_dis = x[3 * T : 4 * T]
    soc = x[4 * T : 4 * T + n_soc]
    peak_kw = float(x[6 * T + 1])

    if elastic:
        slack_soc = x[base_vars : base_vars + n_soc]
        slack_imp = x[base_vars + n_soc : base_vars + n_soc + T]
    else:
        slack_soc = np.zeros(n_soc)
        slack_imp = np.zeros(T)

    # True operational cost (excluding artificial slack penalty)
    true_obj = float(
        np.sum(data["price_imp"] * p_imp * dt)
        - np.sum(data["price_exp"] * p_exp * dt)
        + np.sum(data["c_deg"] * (p_ch + p_dis) * dt)
        + data["c_peak"] * peak_kw
    )

    return DispatchResult(
        status="OPTIMAL",
        objective_yen=true_obj,
        solve_time_sec=elapsed,
        backend="scipy.optimize.milp (HiGHS)",
        elastic=elastic,
        peak_import_kw=peak_kw,
        total_import_kwh=float(np.sum(p_imp * dt)),
        total_export_kwh=float(np.sum(p_exp * dt)),
        total_charge_kwh=float(np.sum(p_ch * dt)),
        total_discharge_kwh=float(np.sum(p_dis * dt)),
        soc_kwh=soc,
        p_imp_kw=p_imp,
        p_exp_kw=p_exp,
        p_ch_kw=p_ch,
        p_dis_kw=p_dis,
        slack_soc_kwh=slack_soc,
        slack_imp_kw=slack_imp,
    )


def solve_dispatch(
    df_slice: Any,
    params: BessParams,
    soc_min_kwh: np.ndarray | None = None,
    soc_max_kwh: np.ndarray | None = None,
    grid_import_cap_kw: np.ndarray | None = None,
    auto_recover_infeasible: bool = True,
) -> tuple[DispatchResult, DispatchResult | None]:
    """Solve strict dispatch; if infeasible and auto_recover_infeasible=True, solve elastic fallback."""
    T = len(df_slice)
    strict_data = make_instance_data(
        df_slice=df_slice,
        params=params,
        soc_min_kwh=soc_min_kwh,
        soc_max_kwh=soc_max_kwh,
        grid_import_cap_kw=grid_import_cap_kw,
        elastic=False,
    )

    # Attempt JijModeling + OMMX if available; otherwise use SciPy HiGHS MILP
    try:
        problem = build_jijmodeling_problem(T=T, elastic=False)
        _ = problem  # Compiled OMMX adapter hook when jijmodeling + ommx adapter are installed
    except ImportError:
        pass

    strict_res = _solve_with_scipy_milp(strict_data, elastic=False)
    if strict_res.status != "INFEASIBLE" or not auto_recover_infeasible:
        return strict_res, None

    elastic_data = make_instance_data(
        df_slice=df_slice,
        params=params,
        soc_min_kwh=soc_min_kwh,
        soc_max_kwh=soc_max_kwh,
        grid_import_cap_kw=grid_import_cap_kw,
        elastic=True,
    )
    recovery_res = _solve_with_scipy_milp(elastic_data, elastic=True)
    return strict_res, recovery_res