"""All backends implement the same model, so they must reach the same optimal cost.

This is the test that caught a sign error in the original SciPy power balance.
"""

import importlib.util

import numpy as np
import pytest

from core_model.solver_runner import solve_dispatch

SCIP_AVAILABLE = importlib.util.find_spec("ommx_pyscipopt_adapter") is not None


@pytest.mark.parametrize(
    "solver",
    ["scipy", pytest.param("scip", marks=pytest.mark.skipif(not SCIP_AVAILABLE, reason="SCIP adapter not installed"))],
)
@pytest.mark.parametrize("scenario", ["day", "week", "typhoon"])
def test_backends_agree(solver, scenario, request, params, typhoon_overrides):
    if scenario == "typhoon":
        window, kw = request.getfixturevalue("day"), typhoon_overrides
    else:
        window, kw = request.getfixturevalue(scenario), {}

    ref_s, ref_r = solve_dispatch(window, params, solver="highs", **kw)
    oth_s, oth_r = solve_dispatch(window, params, solver=solver, **kw)

    assert ref_s.status == oth_s.status
    ref, oth = (ref_r or ref_s), (oth_r or oth_s)
    assert oth.objective_yen == pytest.approx(ref.objective_yen, rel=1e-6)
    assert oth.peak_import_kw == pytest.approx(ref.peak_import_kw, abs=1e-3)
    assert np.sum(oth.slack_imp_kw) == pytest.approx(np.sum(ref.slack_imp_kw), abs=1e-3)
