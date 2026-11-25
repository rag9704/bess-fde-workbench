import numpy as np
import pytest

from core_model.evaluation import (
    Schedule,
    compare_policies,
    cost_breakdown,
    no_battery_schedule,
    rule_based_schedule,
)
from core_model.solver_runner import solve_dispatch

TOL = 1e-6


def _check_physics(w, s, params):
    residual = w["load_kw"] + s.p_ch_kw + s.p_exp_kw - w["solar_kw"] - s.p_dis_kw - s.p_imp_kw
    if s.curtail_kw is not None:
        residual = residual + s.curtail_kw
    assert np.abs(residual).max() < 1e-6
    assert s.p_exp_kw.max() <= params.grid_export_cap_kw + TOL
    assert np.minimum(s.p_ch_kw, s.p_dis_kw).max() < TOL


def test_cost_breakdown_matches_solver_objective(week, params):
    r, _ = solve_dispatch(week, params)
    cb = cost_breakdown(week, r.p_imp_kw, r.p_exp_kw, r.p_ch_kw, r.p_dis_kw, r.soc_kwh, params)
    assert cb.total_cost_yen == pytest.approx(r.objective_yen, rel=1e-9)


def test_no_battery_schedule(day, params):
    s = no_battery_schedule(day, params)
    _check_physics(day, s, params)
    assert s.p_ch_kw.sum() == 0 and s.p_dis_kw.sum() == 0


def test_rule_based_schedule_is_physical_and_does_not_raise_peak(week, params):
    s = rule_based_schedule(week, params)
    nb = no_battery_schedule(week, params)
    _check_physics(week, s, params)
    soc_floor = params.default_soc_min_frac * params.capacity_kwh
    assert s.soc_kwh.min() >= soc_floor - TOL and s.soc_kwh.max() <= params.capacity_kwh + TOL
    expected = s.soc_kwh[:-1] + params.charge_eff * s.p_ch_kw - s.p_dis_kw / params.discharge_eff
    assert np.abs(s.soc_kwh[1:] - expected).max() < 1e-6
    assert s.p_imp_kw.max() <= nb.p_imp_kw.max() + TOL


@pytest.mark.parametrize("window_name", ["day", "week"])
def test_optimizer_beats_baselines(window_name, request, params):
    w = request.getfixturevalue(window_name)
    r, _ = solve_dispatch(w, params)
    opt = Schedule("Optimized", r.p_imp_kw, r.p_exp_kw, r.p_ch_kw, r.p_dis_kw, r.soc_kwh)
    table = compare_policies(w, params, [no_battery_schedule(w, params), rule_based_schedule(w, params), opt])

    # Leaving the battery idle is feasible for the optimizer, so it can never do worse (raw cost).
    assert table.loc["Optimized", "total_cost_yen"] <= table.loc["No battery", "total_cost_yen"] + 1e-6
    assert table.loc["Optimized", "savings_yen"] > 0
    assert table["savings_yen"].idxmax() == "Optimized"
    assert table.loc["No battery", "savings_yen"] == 0
