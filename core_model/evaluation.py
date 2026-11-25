"""Cost accounting and baseline policies, so optimizer value is measurable.

Every schedule (optimized or baseline) is scored with the *same* cost function:

    energy cost - export revenue + battery wear + demand charge on peak import

and compared with a stored-energy adjustment: if a policy ends the horizon with
less energy in the battery than it started with, it "borrowed" that energy, so
the change is valued at the horizon's average import price.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .formulation import BessParams


@dataclass(frozen=True)
class Schedule:
    """Hourly power flows of a policy. Arrays have length T; `soc_kwh` has T+1."""

    name: str
    p_imp_kw: np.ndarray
    p_exp_kw: np.ndarray
    p_ch_kw: np.ndarray
    p_dis_kw: np.ndarray
    soc_kwh: np.ndarray
    curtail_kw: np.ndarray | None = None


@dataclass(frozen=True)
class CostBreakdown:
    energy_cost_yen: float
    export_revenue_yen: float
    wear_cost_yen: float
    peak_import_kw: float
    demand_charge_yen: float
    total_cost_yen: float
    soc_change_kwh: float
    stored_energy_adj_yen: float
    adjusted_cost_yen: float
    import_cap_violation_kwh: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def cost_breakdown(
    df: pd.DataFrame,
    p_imp_kw: np.ndarray,
    p_exp_kw: np.ndarray,
    p_ch_kw: np.ndarray,
    p_dis_kw: np.ndarray,
    soc_kwh: np.ndarray,
    params: BessParams,
    grid_import_cap_kw: np.ndarray | None = None,
) -> CostBreakdown:
    """Score a schedule with the optimizer's cost function (without slack penalties)."""
    dt = params.dt_hours
    price_imp = df["import_price_yen_kwh"].to_numpy(dtype=float)
    price_exp = df["export_price_yen_kwh"].to_numpy(dtype=float)

    energy = float(np.sum(price_imp * p_imp_kw) * dt)
    revenue = float(np.sum(price_exp * p_exp_kw) * dt)
    wear = float(params.degradation_cost_yen_kwh * np.sum(p_ch_kw + p_dis_kw) * dt)
    peak = float(np.max(p_imp_kw)) if len(p_imp_kw) else 0.0
    demand = params.peak_demand_charge_yen_kw * peak
    total = energy - revenue + wear + demand

    soc_change = float(soc_kwh[-1] - soc_kwh[0]) if len(soc_kwh) else 0.0
    # Energy left in (or taken out of) the battery, valued at the average import price
    # as delivered energy (after discharge losses).
    adj = -soc_change * float(np.mean(price_imp)) * params.discharge_eff
    cap = (
        np.full(len(p_imp_kw), params.grid_import_cap_kw)
        if grid_import_cap_kw is None
        else np.asarray(grid_import_cap_kw, dtype=float)
    )
    violation = float(np.sum(np.maximum(p_imp_kw - cap, 0.0)) * dt)

    return CostBreakdown(
        energy_cost_yen=energy,
        export_revenue_yen=revenue,
        wear_cost_yen=wear,
        peak_import_kw=peak,
        demand_charge_yen=demand,
        total_cost_yen=total,
        soc_change_kwh=soc_change,
        stored_energy_adj_yen=adj,
        adjusted_cost_yen=total + adj,
        import_cap_violation_kwh=violation,
    )


def _settle_with_grid(net_kw: float, params: BessParams) -> tuple[float, float, float]:
    """Given net site demand after battery (kW, negative = surplus), return (import, export, curtail)."""
    if net_kw >= 0:
        return net_kw, 0.0, 0.0
    surplus = -net_kw
    exp = min(surplus, params.grid_export_cap_kw)
    return 0.0, exp, surplus - exp


def no_battery_schedule(df: pd.DataFrame, params: BessParams) -> Schedule:
    """Solar + grid only. Surplus PV is exported up to the export cap, then curtailed."""
    T = len(df)
    net = df["load_kw"].to_numpy(dtype=float) - df["solar_kw"].to_numpy(dtype=float)
    imp, exp, cur = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        imp[t], exp[t], cur[t] = _settle_with_grid(net[t], params)
    soc = np.full(T + 1, params.init_soc_kwh)
    return Schedule("No battery", imp, exp, np.zeros(T), np.zeros(T), soc, cur)


def rule_based_schedule(
    df: pd.DataFrame,
    params: BessParams,
    cheap_hours: int = 6,
    expensive_hours: int = 6,
) -> Schedule:
    """A typical hand-tuned operating rule (what a site would run without an optimizer).

    Per calendar day:
      * always store surplus solar,
      * charge from the grid during the `cheap_hours` cheapest hours, but never push grid
        import above the site's existing (no-battery) peak, so charging does not create a
        new demand-charge peak,
      * discharge to cover site load during the `expensive_hours` most expensive hours,
      * never export from the battery, never go below the default reserve floor.
    """
    T = len(df)
    dt = params.dt_hours
    load = df["load_kw"].to_numpy(dtype=float)
    solar = df["solar_kw"].to_numpy(dtype=float)
    price = df["import_price_yen_kwh"].to_numpy(dtype=float)
    days = pd.to_datetime(df["timestamp"]).dt.date.to_numpy()

    cheap = np.zeros(T, dtype=bool)
    pricey = np.zeros(T, dtype=bool)
    for d in np.unique(days):
        idx = np.where(days == d)[0]
        order = idx[np.argsort(price[idx])]
        cheap[order[:cheap_hours]] = True
        pricey[order[::-1][:expensive_hours]] = True
    cheap &= ~pricey

    soc_floor = params.default_soc_min_frac * params.capacity_kwh
    peak_target = min(float(np.max(load - solar)), params.grid_import_cap_kw)
    imp, exp, ch, dis, cur = (np.zeros(T) for _ in range(5))
    soc = np.zeros(T + 1)
    soc[0] = params.init_soc_kwh

    for t in range(T):
        net = load[t] - solar[t]
        room_kwh = params.capacity_kwh - soc[t]
        avail_kwh = max(soc[t] - soc_floor, 0.0)

        if net < 0:  # surplus solar -> charge first
            ch[t] = min(-net, params.max_charge_kw, room_kwh / (params.charge_eff * dt))
        elif cheap[t]:  # cheap grid hour -> charge without raising the site peak
            headroom = max(peak_target - net, 0.0)
            ch[t] = min(params.max_charge_kw, headroom, room_kwh / (params.charge_eff * dt))
        elif pricey[t]:  # expensive hour -> discharge to serve load only
            dis[t] = min(net, params.max_discharge_kw, avail_kwh * params.discharge_eff / dt)

        imp[t], exp[t], cur[t] = _settle_with_grid(net + ch[t] - dis[t], params)
        soc[t + 1] = soc[t] + params.charge_eff * ch[t] * dt - dis[t] * dt / params.discharge_eff

    return Schedule("Rule-based (cheap/expensive hours)", imp, exp, ch, dis, soc, cur)


def compare_policies(
    df: pd.DataFrame,
    params: BessParams,
    schedules: list[Schedule],
    grid_import_cap_kw: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score schedules side by side. Savings are vs the first schedule (the reference)."""
    rows = []
    for s in schedules:
        cb = cost_breakdown(df, s.p_imp_kw, s.p_exp_kw, s.p_ch_kw, s.p_dis_kw, s.soc_kwh, params, grid_import_cap_kw)
        rows.append({"policy": s.name, **cb.as_dict()})
    out = pd.DataFrame(rows).set_index("policy")
    ref = out["adjusted_cost_yen"].iloc[0]
    out["savings_yen"] = ref - out["adjusted_cost_yen"]
    out["savings_pct"] = 100.0 * out["savings_yen"] / ref if ref else 0.0
    return out


def _yen(v: float) -> str:
    return f"-¥{-v:,.0f}" if v < 0 else f"¥{v:,.0f}"


def format_comparison(table: pd.DataFrame) -> str:
    """Human-readable comparison table for the CLI."""
    lines = [
        f"{'Policy':<36}{'Cost (adj.)':>14}{'Peak kW':>10}{'Demand chg':>14}{'Savings':>14}{'%':>8}",
        "-" * 95,
    ]
    for name, r in table.iterrows():
        lines.append(
            f"{name:<36}{_yen(r.adjusted_cost_yen):>14}{r.peak_import_kw:>10.1f}"
            f"{_yen(r.demand_charge_yen):>14}{_yen(r.savings_yen):>14}{r.savings_pct:>7.1f}%"
        )
    if (table["import_cap_violation_kwh"] > 1e-6).any():
        bad = table.index[table["import_cap_violation_kwh"] > 1e-6].tolist()
        lines.append(f"Note: grid import cap exceeded by: {', '.join(bad)}")
    lines.append("Cost (adj.) = energy - export revenue + wear + demand charge, adjusted for change in stored energy.")
    return "\n".join(lines)
