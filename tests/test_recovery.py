import numpy as np

from ai_harness.diagnostic_agent import build_operator_brief
from core_model.solver_runner import solve_dispatch


def test_infeasible_scenario_is_recovered_with_located_violations(day, params, typhoon_overrides):
    strict, recovery = solve_dispatch(day, params, **typhoon_overrides)
    assert strict.status == "INFEASIBLE"
    assert recovery is not None and recovery.status == "OPTIMAL" and recovery.elastic
    assert recovery.slack_imp_kw.sum() > 0

    timestamps = day["timestamp"].astype(str).tolist()
    notes = recovery.infeasibility_summary(timestamps)
    assert any("Grid import cap exceeded" in n and "2026-04-01" in n for n in notes)

    brief = build_operator_brief(strict, recovery, timestamps)
    assert "STRICT INFEASIBLE" in brief and "Elastic Recovery: OPTIMAL" in brief


def test_recovery_can_be_disabled(day, params, typhoon_overrides):
    strict, recovery = solve_dispatch(day, params, auto_recover_infeasible=False, **typhoon_overrides)
    assert strict.status == "INFEASIBLE" and recovery is None


def test_feasible_scenario_does_not_run_recovery(day, params):
    strict, recovery = solve_dispatch(day, params, grid_import_cap_kw=np.full(24, 1000.0))
    assert strict.status == "OPTIMAL" and recovery is None
