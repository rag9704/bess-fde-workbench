from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ai_harness.diagnostic_agent import build_operator_brief
from core_model.solver_runner import SolverOptions, solve_dispatch

EXACT = SolverOptions(mip_rel_gap=1e-9)


def negative_midday_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = pd.to_datetime(out["timestamp"]).dt.hour
    out.loc[(hour >= 10) & (hour < 14), "import_price_yen_kwh"] = -5.0
    return out


def simultaneous(r) -> int:
    return int((np.minimum(r.p_ch_kw, r.p_dis_kw) > 1e-6).sum())


@pytest.mark.parametrize("solver", ["highs", "scip", "scipy"])
def test_lp_first_is_exact_when_lp_is_milp_feasible(week, params, solver):
    exact, _ = solve_dispatch(week, params, solver=solver, options=EXACT)
    fast, _ = solve_dispatch(week, params, solver=solver, options=SolverOptions(lp_first=True))
    assert "LP optimum is MILP-feasible" in fast.backend
    assert fast.status == "OPTIMAL" and not fast.relaxed and simultaneous(fast) == 0
    assert fast.objective_yen == pytest.approx(exact.objective_yen, rel=1e-7)


def burn_scenario(params):
    """One step, full battery, negative price, no export, no demand charge, no wear.

    Importing more is profitable, but the battery is full, so the only way to absorb energy is to
    charge and discharge at once and lose it as heat. The LP relaxation (z in [0, 1]) can do that;
    a real battery (the MILP) cannot, and with a single step it cannot alternate across steps either.
    """
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-01 12:00"]),
            "load_kw": [100.0],
            "solar_kw": [0.0],
            "import_price_yen_kwh": [-100.0],
            "export_price_yen_kwh": [0.0],
        }
    )
    p = replace(
        params,
        init_soc_kwh=params.capacity_kwh,
        end_soc_min_kwh=0.9 * params.capacity_kwh,
        grid_export_cap_kw=0.0,
        peak_demand_charge_yen_kw=0.0,
        degradation_cost_yen_kwh=0.0,
    )
    return df, p


def test_lp_first_falls_back_when_lp_cheats(params):
    df, p = burn_scenario(params)
    lp, _ = solve_dispatch(df, p, options=SolverOptions(relax_integrality=True))
    assert simultaneous(lp) > 0  # the relaxation is not a valid schedule here

    fast, _ = solve_dispatch(df, p, options=SolverOptions(lp_first=True))
    exact, _ = solve_dispatch(df, p, options=EXACT)
    assert "fell back to MILP" in fast.backend
    assert simultaneous(fast) == 0
    assert fast.objective_yen == pytest.approx(exact.objective_yen, rel=1e-6)
    assert lp.objective_yen < exact.objective_yen - 1.0  # a strict integrality gap
    assert any(k.startswith("lp_") for k in fast.timings)


def test_relaxation_is_a_lower_bound(week, params):
    df = negative_midday_prices(week)
    p = replace(params, degradation_cost_yen_kwh=0.0)
    lp, _ = solve_dispatch(df, p, options=SolverOptions(relax_integrality=True))
    milp, _ = solve_dispatch(df, p, options=EXACT)
    assert lp.relaxed and "[LP relaxation]" in lp.backend and lp.mip_gap is None
    assert lp.objective_yen <= milp.objective_yen + 1e-6


def test_gap_is_reported(week, params):
    r, _ = solve_dispatch(week, params, options=SolverOptions(mip_rel_gap=0.05))
    assert r.usable and r.mip_gap is not None and 0 <= r.mip_gap <= 0.05 + 1e-9


def test_no_solution_brief(day, params):
    r, rec = solve_dispatch(day, params)
    no_solution = replace(r, status="NO_SOLUTION", objective_yen=None)
    brief = build_operator_brief(no_solution, None)
    assert "NO_SOLUTION" in brief and "time limit" in brief and "conflicted" not in brief


def test_feasible_brief_mentions_gap(day, params):
    r, _ = solve_dispatch(day, params)
    feasible = replace(r, status="FEASIBLE", mip_gap=0.0123)
    assert "1.23% of optimal" in build_operator_brief(feasible, None)
