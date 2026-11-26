"""Reusable components: equivalence with the hand-written BESS model, an independent DP cross-check,
physics of the factory example, and config validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from components import ConfigError, load_site, site_from_dict, solve_site
from core_model.formulation import BessParams
from core_model.solver_runner import SolverOptions, solve_dispatch

ROOT = Path(__file__).resolve().parent.parent
EXACT = SolverOptions(mip_rel_gap=1e-9)
TOL = 1e-5


@pytest.fixture(scope="module")
def bess_site():
    return load_site(ROOT / "sites" / "tokyo_office_bess.yaml")


@pytest.fixture(scope="module")
def factory():
    site = load_site(ROOT / "sites" / "factory_inventory.yaml")
    return site, solve_site(site, site.load_data())


# --------------------------------------------------------------------------- same answer as core_model


@pytest.mark.parametrize("steps", [24, 168])
def test_bess_site_matches_hand_written_model(bess_site, steps):
    window = bess_site.load_data().iloc[:steps].reset_index(drop=True)
    ref, _ = solve_dispatch(window, BessParams(), options=EXACT)
    got = solve_site(bess_site, window)
    assert got.status == "OPTIMAL"
    assert got.objective == pytest.approx(ref.objective_yen, rel=1e-8)
    assert got.costs["demand_charge"] == pytest.approx(1400 * ref.p_imp_kw.max(), rel=1e-6)


def test_bess_site_on_scip(bess_site):
    window = bess_site.load_data().iloc[:24].reset_index(drop=True)
    assert solve_site(bess_site, window, solver="scip").objective == pytest.approx(
        solve_site(bess_site, window).objective, rel=1e-8
    )


def test_rolling_state_matches_prior_peak_and_initial_soc(bess_site):
    window = bess_site.load_data().iloc[24:48].reset_index(drop=True)
    params = BessParams(init_soc_kwh=1500.0)
    ref, _ = solve_dispatch(window, params, options=EXACT, prior_peak_kw=950.0)
    got = solve_site(bess_site, window, state={"battery.level": 1500.0, "demand_charge.peak": 950.0})
    assert got.objective == pytest.approx(ref.objective_yen, rel=1e-8)
    assert got.get("battery.level")[0] == pytest.approx(1500.0)


# --------------------------------------------------------------------------- independent check: Wagner-Whitin


def wagner_whitin(demand: np.ndarray, setup: float, hold: float) -> float:
    """Exact DP for uncapacitated lot sizing (produce in period i to cover demand i..j-1)."""
    T = len(demand)
    best = np.full(T + 1, np.inf)
    best[0] = 0.0
    for j in range(1, T + 1):
        for i in range(j):
            carry = sum(hold * demand[k] * (k - i) for k in range(i, j))
            cost = best[i] + (setup if demand[i:j].sum() > 0 else 0.0) + carry
            best[j] = min(best[j], cost)
    return float(best[T])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lot_sizing_matches_wagner_whitin_dp(seed):
    rng = np.random.default_rng(seed)
    demand = rng.integers(0, 60, size=10).astype(float)
    setup, hold = 250.0, 2.0
    site = site_from_dict(
        {
            "site": "lot_sizing",
            "components": [
                {"type": "node", "name": "n"},
                {"type": "profile", "name": "demand", "node": "n", "direction": "out", "column": "d"},
                {"type": "producer", "name": "plant", "node": "n", "capacity": 1e4, "setup_cost": setup},
                {"type": "storage", "name": "stock", "node": "n", "capacity": 1e5, "holding_cost": hold},
            ],
        }
    )
    got = solve_site(site, pd.DataFrame({"d": demand}))
    # Our holding cost is charged on end-of-period stock, exactly the DP's carrying cost.
    assert got.objective == pytest.approx(wagner_whitin(demand, setup, hold), abs=1e-6)


# --------------------------------------------------------------------------- factory physics


def test_factory_balance_and_inventory(factory):
    site, r = factory
    demand, unmet = r.get("customers.value"), r.get("customers.unmet")
    made, bought, level = r.get("line.output"), r.get("supplier.buy"), r.get("warehouse.level")
    fill, draw = r.get("warehouse.in"), r.get("warehouse.out")
    assert np.allclose(made + bought + draw, demand - unmet + fill, atol=TOL)
    assert np.allclose(level[1:], level[:-1] + fill - draw, atol=TOL)
    assert level[0] == pytest.approx(150) and level[-1] >= 150 - TOL
    assert (level >= 50 - TOL).all() and (level <= 600 + TOL).all()


def test_factory_setup_logic(factory):
    _, r = factory
    made, on = r.get("line.output"), np.round(r.get("line.on"))
    assert (made[on == 0] <= TOL).all()  # no output without a setup
    assert (made[on == 1] >= 60 - TOL).all() and (made <= 140 + TOL).all()
    assert r.costs["line"] == pytest.approx(float((r.get("line.unit_cost") * made).sum() + 3000 * on.sum()), rel=1e-9)
    assert (r.get("supplier.buy") <= 80 + TOL).all()


def test_factory_costs_add_up(factory):
    _, r = factory
    assert r.status == "OPTIMAL" and r.objective == pytest.approx(sum(r.costs.values()))


# --------------------------------------------------------------------------- config validation


def base():
    return [
        {"type": "node", "name": "n"},
        {"type": "profile", "name": "d", "node": "n", "direction": "out", "column": "x"},
        {"type": "market", "name": "m", "node": "n", "buy_price": 1.0},
    ]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.append({"type": "battery", "name": "b"}), "unknown type"),
        (lambda c: c[2].update(colour="red"), "unknown field"),
        (lambda c: c.append({"type": "node", "name": "n"}), "duplicate"),
        (lambda c: c[1].update(node="nowhere"), "unknown node"),
        (lambda c: c.append({"type": "peak_charge", "name": "p", "flow": "grid.buy", "rate": 1}), "unknown flow"),
        (lambda c: c[1].update(column="missing"), "column 'missing'"),
        (lambda c: c[0].update(name="has space"), "identifier"),
    ],
)
def test_config_errors_are_specific(mutate, match):
    comps = base()
    mutate(comps)
    with pytest.raises(ConfigError, match=match):
        site = site_from_dict({"site": "s", "components": comps})
        solve_site(site, pd.DataFrame({"x": [1.0, 2.0]}))


def test_cli_runs(capsys):
    from components.__main__ import main

    assert main([str(ROOT / "sites" / "factory_inventory.yaml"), "--steps", "14", "--show", "line.output"]) == 0
    out = capsys.readouterr().out
    assert "factory_inventory: OPTIMAL" in out and "line.output" in out
