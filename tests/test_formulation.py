"""Physical sanity of optimal schedules (the bess-model-verifier checklist, automated)."""

import numpy as np
import pytest

from core_model.formulation import build_problem, make_instance_data
from core_model.solver_runner import solve_dispatch

TOL = 1e-5


@pytest.fixture(scope="module", params=["day", "week"])
def solved(request, params):
    window = request.getfixturevalue(request.param)
    strict, _ = solve_dispatch(window, params)
    assert strict.status == "OPTIMAL"
    return window, strict


def test_power_balance(solved):
    w, r = solved
    residual = w["load_kw"] + r.p_ch_kw + r.p_exp_kw - w["solar_kw"] - r.p_dis_kw - r.p_imp_kw
    assert np.abs(residual).max() < TOL


def test_soc_dynamics_and_bounds(solved, params):
    _, r = solved
    expected = r.soc_kwh[:-1] + params.charge_eff * r.p_ch_kw - r.p_dis_kw / params.discharge_eff
    assert np.abs(r.soc_kwh[1:] - expected).max() < TOL
    assert r.soc_kwh[0] == pytest.approx(params.init_soc_kwh)
    assert r.soc_kwh[-1] >= params.end_soc_min_kwh - TOL
    assert r.soc_kwh.min() >= params.default_soc_min_frac * params.capacity_kwh - TOL
    assert r.soc_kwh.max() <= params.capacity_kwh + TOL


def test_no_simultaneous_charge_discharge(solved):
    _, r = solved
    assert np.minimum(r.p_ch_kw, r.p_dis_kw).max() < TOL


def test_power_limits(solved, params):
    _, r = solved
    assert r.p_ch_kw.max() <= params.max_charge_kw + TOL
    assert r.p_dis_kw.max() <= params.max_discharge_kw + TOL
    assert r.p_exp_kw.max() <= params.grid_export_cap_kw + TOL
    assert r.p_imp_kw.max() <= params.grid_import_cap_kw + TOL
    assert r.peak_import_kw == pytest.approx(r.p_imp_kw.max(), abs=1e-4)


def test_strict_has_no_slack(solved):
    _, r = solved
    assert r.slack_imp_kw.max() < TOL and r.slack_soc_kwh.max() < TOL


def test_model_is_built_once_and_reused():
    assert build_problem() is build_problem()


def test_instance_data_length_checks(day, params):
    with pytest.raises(ValueError, match="soc_min_kwh"):
        make_instance_data(day, params, soc_min_kwh=np.zeros(5))
    with pytest.raises(ValueError, match="grid_import_cap_kw"):
        make_instance_data(day, params, grid_import_cap_kw=np.zeros(5))


def test_soc_max_override_is_respected(day, params):
    cap = np.full(len(day), 1200.0)
    strict, _ = solve_dispatch(day, params, soc_max_kwh=cap)
    assert strict.soc_kwh[: len(day)].max() <= 1200.0 + TOL
