"""Forecasters (no look-ahead, sensible accuracy) and the rolling re-planning simulator (physics, guard)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core_model.formulation import BessParams
from forecasting.forecasters import (
    TARGETS,
    GBMForecaster,
    PerfectForecaster,
    PersistenceForecaster,
    site_prices,
)
from forecasting.mpc import guard, run_mpc
from service.pipeline import DEFAULT_DATA, load_data


@pytest.fixture(scope="module")
def actual():
    return load_data(DEFAULT_DATA)


@pytest.fixture(scope="module")
def gbm(actual):
    return GBMForecaster(max_iter=60).fit(actual.iloc[: 14 * 24])


def test_site_prices_match_dataset(actual):
    imp, exp = site_prices(actual["tokyo_price_yen_kwh"].to_numpy())
    assert np.allclose(imp, actual["import_price_yen_kwh"], atol=0.01)
    assert np.allclose(exp, actual["export_price_yen_kwh"], atol=0.01)


def test_persistence_is_yesterday(actual):
    hist, fut = actual.iloc[:100], pd.DatetimeIndex(actual["timestamp"].iloc[100:124])
    fc = PersistenceForecaster().forecast(hist, fut)
    assert np.allclose(fc["load_kw"], actual["load_kw"].iloc[76:100])


def test_forecasts_cannot_see_the_future(actual, gbm):
    """Changing everything after the forecast origin must not change the forecast."""
    i = 20 * 24 + 7
    fut = pd.DatetimeIndex(actual["timestamp"].iloc[i : i + 24])
    tampered = actual.copy()
    tampered.loc[i:, list(TARGETS)] = 9999.0
    a = gbm.forecast(actual.iloc[:i], fut)
    b = gbm.forecast(tampered.iloc[:i], fut)
    pd.testing.assert_frame_equal(a, b)
    assert (a["solar_kw"] >= 0).all() and len(a) == 24


def test_gbm_choice_is_validated(gbm):
    assert set(gbm.choice) == set(TARGETS)
    for col, scores in gbm.validation_mae.items():
        best = "gbm" if scores["gbm"] < scores["persistence"] else "persistence"
        assert gbm.choice[col] == best


def test_guard_cuts_charge_before_discharging():
    p = BessParams()
    # net 900 kW, planned charge 200 kW, target 1000 kW -> cut charge by 100
    assert guard(900.0, 200.0, 0.0, 1000.0, 1000.0, 200.0, p) == (100.0, 0.0)
    # net 1100 kW, no charge, target 1000 -> discharge 100
    assert guard(1100.0, 0.0, 0.0, 1000.0, 1000.0, 200.0, p) == (0.0, 100.0)
    # battery at its floor -> nothing left to discharge
    assert guard(1100.0, 0.0, 0.0, 1000.0, 200.0, 200.0, p) == (0.0, 0.0)
    # discharge limited by inverter power
    assert guard(2000.0, 0.0, 0.0, 1000.0, 2000.0, 200.0, p)[1] == pytest.approx(p.max_discharge_kw)


@pytest.mark.parametrize("guard_on", [True, False])
def test_mpc_physics(actual, guard_on):
    params = BessParams()
    r = run_mpc(actual, params, PersistenceForecaster(), 14 * 24, 24, solver="scipy", peak_guard=guard_on)
    s = r.schedule
    window = actual.iloc[14 * 24 : 15 * 24].reset_index(drop=True)
    net = window["load_kw"] - window["solar_kw"] + s.p_ch_kw - s.p_dis_kw
    assert np.allclose(s.p_imp_kw - s.p_exp_kw - s.curtail_kw, net, atol=1e-6)
    assert np.allclose(s.soc_kwh[1:], s.soc_kwh[:-1] + 0.94 * s.p_ch_kw - s.p_dis_kw / 0.94, atol=1e-6)
    assert (s.soc_kwh >= -1e-6).all() and (s.soc_kwh <= params.capacity_kwh + 1e-6).all()
    assert (np.minimum(s.p_ch_kw, s.p_dis_kw) <= 1e-6).all()
    assert (s.p_exp_kw <= params.grid_export_cap_kw + 1e-6).all()
    assert r.solves == 24 and r.fallback_steps == 0


def test_perfect_forecast_mpc_is_close_to_hindsight(actual):
    """With no forecast error the only loss is the rolling horizon: small, and never better than hindsight."""
    from core_model.evaluation import cost_breakdown
    from core_model.solver_runner import solve_dispatch

    params = replace(BessParams(), end_soc_min_kwh=800.0)
    start, steps = 14 * 24, 48
    window = actual.iloc[start : start + steps].reset_index(drop=True)
    r = run_mpc(actual, params, PerfectForecaster(actual), start, steps, solver="scipy")
    hs, _ = solve_dispatch(window, params, solver="scipy")
    hs_cost = cost_breakdown(window, hs.p_imp_kw, hs.p_exp_kw, hs.p_ch_kw, hs.p_dis_kw, hs.soc_kwh, params)
    assert r.forecast_mae == {k: 0.0 for k in TARGETS}
    assert r.cost.adjusted_cost_yen >= hs_cost.adjusted_cost_yen - 1.0
    assert r.cost.adjusted_cost_yen <= hs_cost.adjusted_cost_yen * 1.05


def test_peak_guard_never_raises_the_peak(actual):
    params = BessParams()
    args = (actual, params, PersistenceForecaster(), 14 * 24, 72)
    on = run_mpc(*args, solver="scipy", peak_guard=True)
    off = run_mpc(*args, solver="scipy", peak_guard=False)
    assert on.cost.peak_import_kw <= off.cost.peak_import_kw + 1e-6
    assert on.guard_steps > 0


def test_hourly_only(actual):
    with pytest.raises(ValueError, match="hourly"):
        run_mpc(actual, BessParams(dt_hours=0.5), PersistenceForecaster(), 0, 1)


def test_published_jepx_prices_follow_the_auction_clock(actual):
    from forecasting.forecasters import PublishedPrices

    wrapped = PublishedPrices(PersistenceForecaster(), actual)
    day = actual["timestamp"].dt.normalize()

    def run(i):
        fut = pd.DatetimeIndex(actual["timestamp"].iloc[i : i + 24])
        fc = wrapped.forecast(actual.iloc[:i], fut)
        true = actual["tokyo_price_yen_kwh"].iloc[i : i + 24].to_numpy()
        return fc["tokyo_price_yen_kwh"].to_numpy() == true, day.iloc[i : i + 24].to_numpy() == day.iloc[i]

    # 11:00: today and tomorrow are published, so the whole 24 h horizon uses real prices
    exact, _ = run(5 * 24 + 11)
    assert exact.all()
    # 08:00: today's prices are known, tomorrow's are not yet (auction closes at 10:00)
    exact, today = run(5 * 24 + 8)
    assert exact[today].all()
    persistence_tomorrow = (
        PersistenceForecaster()
        .forecast(actual.iloc[: 5 * 24 + 8], pd.DatetimeIndex(actual["timestamp"].iloc[5 * 24 + 8 : 5 * 24 + 32]))[
            "tokyo_price_yen_kwh"
        ]
        .to_numpy()[~today]
    )
    i = 5 * 24 + 8
    fc = wrapped.forecast(actual.iloc[:i], pd.DatetimeIndex(actual["timestamp"].iloc[i : i + 24]))
    assert np.allclose(fc["tokyo_price_yen_kwh"].to_numpy()[~today], persistence_tomorrow)
