"""Solve the BESS dispatch model with a pluggable backend.

Backends (all consume the same instance data from `make_instance_data`):
  * "highs" (default): JijModeling -> OMMX Instance -> HiGHS via ommx-highs-adapter
  * "scip":            JijModeling -> OMMX Instance -> SCIP via ommx-pyscipopt-adapter
  * "scipy":           hand-built sparse matrix -> scipy.optimize.milp (HiGHS).
                       Independent re-implementation kept as a reference / cross-check.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import logging
import os
import sys
import threading
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .formulation import BessParams, build_problem, make_instance_data

logger = logging.getLogger(__name__)

SOLVERS = ("highs", "scip", "scipy")
BACKEND_LABELS = {
    "highs": "JijModeling + OMMX / HiGHS",
    "scip": "JijModeling + OMMX / SCIP",
    "scipy": "scipy.optimize.milp (reference)",
}


# OPTIMAL: proven optimal (within the requested MIP gap, if any).
# FEASIBLE: a limit (e.g. time) stopped the solver, but it has a usable incumbent.
# NO_SOLUTION: a limit stopped the solver before any feasible schedule was found.
USABLE_STATUSES = ("OPTIMAL", "FEASIBLE")


@dataclass(frozen=True)
class SolverOptions:
    """Knobs for the solution-quality / solve-time trade-off."""

    mip_rel_gap: float | None = None  # stop when (incumbent - bound) / incumbent <= gap
    time_limit_sec: float | None = None  # stop after this long and return the incumbent
    relax_integrality: bool = False  # solve the LP relaxation (binary mode-lock becomes [0, 1])
    # Exact speed-up: solve the LP relaxation first. If its schedule never charges and discharges in
    # the same step it is feasible for the MILP, and since LP cost <= MILP cost it is MILP-optimal.
    # Otherwise fall back to the full MILP. Ignored when relax_integrality=True.
    lp_first: bool = False


SIMULTANEOUS_TOL_KW = 1e-6


@dataclass
class DispatchResult:
    status: str
    objective_yen: float | None  # Operational cost, excluding artificial slack penalties
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
    timings: dict[str, float] = field(default_factory=dict)
    mip_gap: float | None = None  # relative gap reported by the solver (None for LP / unknown)
    relaxed: bool = False

    @property
    def usable(self) -> bool:
        return self.status in USABLE_STATUSES

    def infeasibility_summary(self, timestamps: list[str] | None = None) -> list[str]:
        """Return human-readable constraint violations when elastic slack > 1e-4 is active."""
        notes: list[str] = []
        for idx, val in enumerate(self.slack_imp_kw):
            if val > 1e-4:
                label = timestamps[idx] if timestamps and idx < len(timestamps) else f"t={idx}"
                notes.append(f"Grid import cap exceeded by {val:.1f} kW at {label}")
        for idx, val in enumerate(self.slack_soc_kwh):
            if val > 1e-4:
                # slack_soc is per *state*: state k is the SoC at the start of step k
                label = timestamps[idx] if timestamps and idx < len(timestamps) else f"end of horizon (state {idx})"
                notes.append(f"Minimum SoC reserve shortfall of {val:.1f} kWh at {label}")
        return notes


class Infeasible(Exception):
    """Raised by a backend when the instance has no feasible solution."""


# --------------------------------------------------------------------------- helpers


def _operational_cost(data: dict[str, Any], p_imp, p_exp, p_ch, p_dis, peak) -> float:
    dt = data["dt"]
    return float(
        np.sum(data["price_imp"] * p_imp) * dt
        - np.sum(data["price_exp"] * p_exp) * dt
        + data["c_deg"] * np.sum(p_ch + p_dis) * dt
        + data["c_peak"] * peak
    )


def _make_result(data, elastic, backend, timings, values, status, mip_gap, relaxed) -> DispatchResult:
    dt = data["dt"]
    p_imp, p_exp, p_ch, p_dis = (values[k] for k in ("p_imp", "p_exp", "p_ch", "p_dis"))
    peak = float(values["p_peak"])
    label = BACKEND_LABELS[backend] + (" [LP relaxation]" if relaxed else "")
    return DispatchResult(
        status=status,
        objective_yen=_operational_cost(data, p_imp, p_exp, p_ch, p_dis, peak),
        solve_time_sec=sum(timings.values()),
        backend=label,
        elastic=elastic,
        peak_import_kw=peak,
        total_import_kwh=float(np.sum(p_imp) * dt),
        total_export_kwh=float(np.sum(p_exp) * dt),
        total_charge_kwh=float(np.sum(p_ch) * dt),
        total_discharge_kwh=float(np.sum(p_dis) * dt),
        soc_kwh=values["soc"],
        p_imp_kw=p_imp,
        p_exp_kw=p_exp,
        p_ch_kw=p_ch,
        p_dis_kw=p_dis,
        slack_soc_kwh=values["s_soc_min"],
        slack_imp_kw=values["s_imp_cap"],
        timings=timings,
        mip_gap=mip_gap,
        relaxed=relaxed,
    )


def _empty_result(data, elastic, backend, timings, status: str) -> DispatchResult:
    T = data["T"]
    zt, zs = np.zeros(T), np.zeros(T + 1)
    return DispatchResult(
        status=status,
        objective_yen=None,
        solve_time_sec=sum(timings.values()),
        backend=BACKEND_LABELS[backend],
        elastic=elastic,
        peak_import_kw=0.0,
        total_import_kwh=0.0,
        total_export_kwh=0.0,
        total_charge_kwh=0.0,
        total_discharge_kwh=0.0,
        soc_kwh=zs,
        p_imp_kw=zt,
        p_exp_kw=zt,
        p_ch_kw=zt,
        p_dis_kw=zt,
        slack_soc_kwh=zs,
        slack_imp_kw=zt,
        timings=timings,
    )


class NoSolution(Exception):
    """Raised when a limit stopped the solver before it found any feasible schedule."""


@dataclass
class _RawSolve:
    values: dict[str, np.ndarray]
    timings: dict[str, float]
    status: str
    mip_gap: float | None


# --------------------------------------------------------------------------- OMMX backends

_VAR_NAMES = ("p_imp", "p_exp", "p_ch", "p_dis", "soc", "s_soc_min", "s_imp_cap")


def _extract_values(solution) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in _VAR_NAMES:
        d = solution.extract_decision_variables(name)
        values[name] = np.array([d[k] for k in sorted(d)], dtype=float)
    peak = solution.extract_decision_variables("p_peak")
    values["p_peak"] = np.array(next(iter(peak.values())), dtype=float)
    return values


@contextmanager
def _quiet_c_stdout():
    """HiGHS prints a copyright banner from C code; redirect fd 1 so CLI/API output stays clean."""
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):  # e.g. captured stdout in tests
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


_highs_banner_lock = threading.Lock()
_highs_banner_shown = False


def _new_highs_adapter(instance):
    """Create the adapter (which creates the Highs object), silencing the one-time banner."""
    from ommx_highs_adapter import OMMXHighsAdapter

    global _highs_banner_shown
    # The banner is printed only by the first Highs() in a process. The fd redirect is
    # process-wide, hence the lock (the API solves in worker threads).
    if not _highs_banner_shown:
        with _highs_banner_lock:
            if not _highs_banner_shown:
                with _quiet_c_stdout():
                    adapter = OMMXHighsAdapter(instance)
                _highs_banner_shown = True
                return adapter
    return OMMXHighsAdapter(instance)


def _values_from_highs(instance, h) -> dict[str, np.ndarray]:
    """Read variable values straight from HiGHS' solution vector.

    ommx-highs-adapter's decode()/decode_to_state() index `solution.col_value[i]` inside a loop.
    highspy returns a fresh copy of the whole vector on each `col_value` access, so decoding is
    O(n^2): about 1.7 s for a 30-min month (11.5k variables), far longer than the LP solve.
    The adapter creates columns in `instance.used_decision_variables` order, so one copy is enough.
    """
    col = np.asarray(h.getSolution().col_value, dtype=float)
    grouped: dict[str, list[tuple[tuple[int, ...], float]]] = {}
    for i, var in enumerate(instance.used_decision_variables):
        grouped.setdefault(var.name, []).append((tuple(var.subscripts), col[i]))
    values = {name: np.array([v for _, v in sorted(grouped.get(name, []))], dtype=float) for name in _VAR_NAMES}
    values["p_peak"] = np.array(grouped["p_peak"][0][1], dtype=float)
    return values


def _solve_highs(instance, opts: SolverOptions) -> tuple[dict[str, np.ndarray], str, float | None]:
    import highspy

    adapter = _new_highs_adapter(instance)
    h = adapter.solver_input
    h.setOptionValue("output_flag", False)
    if opts.mip_rel_gap is not None:
        h.setOptionValue("mip_rel_gap", float(opts.mip_rel_gap))
    if opts.time_limit_sec is not None:
        h.setOptionValue("time_limit", float(opts.time_limit_sec))
    if opts.relax_integrality:
        h.setOptionValue("solve_relaxation", True)
    h.run()

    ms = highspy.HighsModelStatus
    model_status = h.getModelStatus()
    info = h.getInfo()
    if model_status in (ms.kInfeasible, ms.kUnboundedOrInfeasible):
        raise Infeasible
    has_solution = info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    if model_status == ms.kOptimal:
        status = "OPTIMAL"
    elif has_solution:
        status = "FEASIBLE"
    else:
        raise NoSolution(h.modelStatusToString(model_status))
    gap = None if opts.relax_integrality or not np.isfinite(info.mip_gap) else float(info.mip_gap)
    return _values_from_highs(instance, h), status, gap


def _solve_scip(instance, opts: SolverOptions) -> tuple[dict[str, np.ndarray], str, float | None]:
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

    adapter = OMMXPySCIPOptAdapter(instance)
    m = adapter.solver_input
    m.hideOutput()
    if opts.mip_rel_gap is not None:
        m.setParam("limits/gap", float(opts.mip_rel_gap))
    if opts.time_limit_sec is not None:
        m.setParam("limits/time", float(opts.time_limit_sec))
    if opts.relax_integrality:
        m.relax()  # turns every integer/binary variable continuous
    m.optimize()

    scip_status = m.getStatus()
    if scip_status in ("infeasible", "inforunbd"):
        raise Infeasible
    if scip_status in ("optimal", "gaplimit"):
        status = "OPTIMAL"
    elif m.getNSols() > 0:
        status = "FEASIBLE"
    else:
        raise NoSolution(scip_status)
    gap = None if opts.relax_integrality else float(m.getGap())
    return _extract_values(adapter.decode(m)), status, gap


def _solve_with_ommx(data: dict[str, Any], solver: str, opts: SolverOptions) -> _RawSolve:
    t0 = time.perf_counter()
    instance = build_problem().eval(data)
    t1 = time.perf_counter()
    try:
        values, status, gap = (_solve_highs if solver == "highs" else _solve_scip)(instance, opts)
    finally:
        t2 = time.perf_counter()
    return _RawSolve(values, {"compile_sec": t1 - t0, "solve_sec": t2 - t1}, status, gap)


# --------------------------------------------------------------------------- SciPy reference


def _solve_with_scipy(data: dict[str, Any], opts: SolverOptions) -> _RawSolve:
    """Independent sparse-matrix implementation of the same model (for cross-checking)."""
    t0 = time.perf_counter()
    T, dt = data["T"], data["dt"]
    S = T + 1
    # Variable offsets
    o = {}
    n = 0
    for name, size in [
        ("p_imp", T),
        ("p_exp", T),
        ("p_ch", T),
        ("p_dis", T),
        ("soc", S),
        ("z_ch", T),
        ("p_peak", 1),
        ("s_soc_min", S),
        ("s_imp_cap", T),
    ]:
        o[name] = n
        n += size

    c = np.zeros(n)
    c[o["p_imp"] : o["p_imp"] + T] = data["price_imp"] * dt
    c[o["p_exp"] : o["p_exp"] + T] = -data["price_exp"] * dt
    c[o["p_ch"] : o["p_ch"] + T] = data["c_deg"] * dt
    c[o["p_dis"] : o["p_dis"] + T] = data["c_deg"] * dt
    c[o["p_peak"]] = data["c_peak"]
    c[o["s_soc_min"] : o["s_soc_min"] + S] = data["m_pen"]
    c[o["s_imp_cap"] : o["s_imp_cap"] + T] = data["m_pen"] * dt

    lb, ub = np.zeros(n), np.zeros(n)
    for name, size, hi in [
        ("p_imp", T, data["p_imp_bound"]),
        ("p_exp", T, data["p_exp_max"]),
        ("p_ch", T, data["p_ch_max"]),
        ("p_dis", T, data["p_dis_max"]),
        ("soc", S, data["cap_kwh"]),
        ("z_ch", T, 1.0),
        ("p_peak", 1, data["p_imp_bound"]),
        ("s_soc_min", S, data["slack_ub_soc"]),
        ("s_imp_cap", T, data["slack_ub_imp"]),
    ]:
        ub[o[name] : o[name] + size] = hi
    integrality = np.zeros(n, dtype=int)
    if not opts.relax_integrality:
        integrality[o["z_ch"] : o["z_ch"] + T] = 1

    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = 0

    def add(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        nonlocal r
        for col, v in entries:
            rows.append(r)
            cols.append(col)
            vals.append(v)
        rlo.append(lo)
        rhi.append(hi)
        r += 1

    soc = o["soc"]
    for t in range(T):
        # load + ch + exp == solar + dis + imp   <=>   imp - exp - ch + dis == load - solar
        rhs = data["load"][t] - data["solar"][t]
        add([(o["p_imp"] + t, 1), (o["p_exp"] + t, -1), (o["p_ch"] + t, -1), (o["p_dis"] + t, 1)], rhs, rhs)
        add(
            [
                (soc + t + 1, 1),
                (soc + t, -1),
                (o["p_ch"] + t, -data["eta_ch"] * dt),
                (o["p_dis"] + t, dt / data["eta_dis"]),
            ],
            0,
            0,
        )
        add([(o["p_ch"] + t, 1), (o["z_ch"] + t, -data["p_ch_max"])], -np.inf, 0)
        add([(o["p_dis"] + t, 1), (o["z_ch"] + t, data["p_dis_max"])], -np.inf, data["p_dis_max"])
        add([(o["p_peak"], 1), (o["p_imp"] + t, -1)], 0, np.inf)
        add([(o["p_imp"] + t, 1), (o["s_imp_cap"] + t, -1)], -np.inf, data["grid_imp_cap"][t])
    add([(soc, 1)], data["soc_init"], data["soc_init"])
    add([(soc + T, 1)], data["soc_end_min"], np.inf)
    for k in range(S):
        add([(soc + k, 1)], -np.inf, data["soc_max"][k])
        add([(soc + k, 1), (o["s_soc_min"] + k, 1)], data["soc_min"][k], np.inf)

    A = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    milp_options: dict[str, float] = {}
    if opts.mip_rel_gap is not None:
        milp_options["mip_rel_gap"] = float(opts.mip_rel_gap)
    if opts.time_limit_sec is not None:
        milp_options["time_limit"] = float(opts.time_limit_sec)
    t1 = time.perf_counter()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A, rlo, rhi),
        options=milp_options or None,
    )
    t2 = time.perf_counter()
    timings = {"compile_sec": t1 - t0, "solve_sec": t2 - t1}
    if res.status == 2:  # infeasible
        raise Infeasible
    if res.x is None:
        raise NoSolution(res.message)
    status = "OPTIMAL" if res.status == 0 else "FEASIBLE"  # 1 = iteration/time limit with incumbent
    gap = getattr(res, "mip_gap", None)
    gap = None if opts.relax_integrality or gap is None or not np.isfinite(gap) else float(gap)
    x = res.x
    values = {
        name: x[o[name] : o[name] + size]
        for name, size in [
            ("p_imp", T),
            ("p_exp", T),
            ("p_ch", T),
            ("p_dis", T),
            ("soc", S),
            ("s_soc_min", S),
            ("s_imp_cap", T),
        ]
    }
    values["p_peak"] = np.array(x[o["p_peak"]])
    return _RawSolve(values, timings, status, gap)


# --------------------------------------------------------------------------- public API


def solve_instance(
    data: dict[str, Any],
    elastic: bool,
    solver: str = "highs",
    options: SolverOptions | None = None,
) -> DispatchResult:
    """Solve one prepared instance with the chosen backend."""
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; choose from {SOLVERS}")
    opts = options or SolverOptions()
    if opts.lp_first and not opts.relax_integrality:
        return _solve_lp_first(data, elastic, solver, opts)
    t0 = time.perf_counter()
    try:
        raw = _solve_with_scipy(data, opts) if solver == "scipy" else _solve_with_ommx(data, solver, opts)
    except Infeasible:
        result = _empty_result(data, elastic, solver, {"total_sec": time.perf_counter() - t0}, "INFEASIBLE")
    except NoSolution:
        result = _empty_result(data, elastic, solver, {"total_sec": time.perf_counter() - t0}, "NO_SOLUTION")
    else:
        result = _make_result(
            data, elastic, solver, raw.timings, raw.values, raw.status, raw.mip_gap, opts.relax_integrality
        )
    logger.info(
        "solve finished",
        extra={
            "solver": solver,
            "elastic": elastic,
            "horizon": data["T"],
            "status": result.status,
            "objective_yen": result.objective_yen,
            "mip_gap": result.mip_gap,
            "solve_time_sec": round(result.solve_time_sec, 4),
        },
    )
    return result


def _solve_lp_first(data: dict[str, Any], elastic: bool, solver: str, opts: SolverOptions) -> DispatchResult:
    lp = solve_instance(data, elastic, solver, replace(opts, relax_integrality=True, lp_first=False))
    if lp.status == "OPTIMAL" and np.minimum(lp.p_ch_kw, lp.p_dis_kw).max(initial=0.0) <= SIMULTANEOUS_TOL_KW:
        # LP optimum is MILP-feasible => MILP-optimal (certificate), no branching needed.
        lp.relaxed = False
        lp.mip_gap = 0.0
        lp.backend = BACKEND_LABELS[solver] + " [LP-first: LP optimum is MILP-feasible]"
        lp.timings = {f"lp_{k}": v for k, v in lp.timings.items()}
        return lp
    milp_result = solve_instance(data, elastic, solver, replace(opts, lp_first=False))
    milp_result.timings = {**{f"lp_{k}": v for k, v in lp.timings.items()}, **milp_result.timings}
    milp_result.solve_time_sec = sum(milp_result.timings.values())
    milp_result.backend += " [LP-first: fell back to MILP]"
    return milp_result


def solve_dispatch(
    df_slice: Any,
    params: BessParams,
    soc_min_kwh: np.ndarray | None = None,
    soc_max_kwh: np.ndarray | None = None,
    grid_import_cap_kw: np.ndarray | None = None,
    auto_recover_infeasible: bool = True,
    solver: str = "highs",
    options: SolverOptions | None = None,
) -> tuple[DispatchResult, DispatchResult | None]:
    """Solve strict dispatch; if infeasible and auto_recover_infeasible=True, solve elastic fallback."""
    kwargs = dict(
        df_slice=df_slice,
        params=params,
        soc_min_kwh=soc_min_kwh,
        soc_max_kwh=soc_max_kwh,
        grid_import_cap_kw=grid_import_cap_kw,
    )
    strict = solve_instance(make_instance_data(**kwargs, elastic=False), False, solver, options)
    if strict.status != "INFEASIBLE" or not auto_recover_infeasible:
        return strict, None
    recovery = solve_instance(make_instance_data(**kwargs, elastic=True), True, solver, options)
    return strict, recovery
