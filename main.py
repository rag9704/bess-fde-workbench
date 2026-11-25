"""Command-line entry point for the BESS + PV dispatch workbench.

Examples:
    python main.py                                   # first 24h of the dataset
    python main.py --start 2026-04-07 --hours 168    # one week from Apr 7
    python main.py --alert "Typhoon warning, cap grid import at 300 kW"
    python main.py --capacity-kwh 3000 --max-charge-kw 750 --out schedule.csv
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_harness.diagnostic_agent import build_operator_brief
from ai_harness.exception_translator import translate_exception_text
from core_model.formulation import BessParams
from core_model.solver_runner import DispatchResult, solve_dispatch

DEFAULT_DATA = Path(__file__).parent / "data" / "april_2026_hourly.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize hourly battery dispatch for a solar + BESS site.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Hourly site CSV (see data/prepare_japan_grid_data.py)")
    p.add_argument("--start", type=str, default=None, help="Start timestamp, e.g. 2026-04-07 or '2026-04-07 06:00'. Default: first row")
    p.add_argument("--hours", type=int, default=24, help="Horizon length in hours (default 24; 168 = week, 720 = month)")
    p.add_argument("--alert", type=str, default=None, help="Operator alert text to translate into reserve / grid-cap overrides")
    p.add_argument("--no-recovery", action="store_true", help="Do not run the elastic (slack) re-solve when strict is infeasible")
    p.add_argument("--out", type=Path, default=None, help="Write the hourly schedule to this CSV")

    g = p.add_argument_group("battery / site parameters (defaults from BessParams)")
    d = BessParams()
    g.add_argument("--capacity-kwh", type=float, default=d.capacity_kwh)
    g.add_argument("--max-charge-kw", type=float, default=d.max_charge_kw)
    g.add_argument("--max-discharge-kw", type=float, default=d.max_discharge_kw)
    g.add_argument("--init-soc-kwh", type=float, default=d.init_soc_kwh)
    g.add_argument("--end-soc-min-kwh", type=float, default=d.end_soc_min_kwh)
    g.add_argument("--grid-import-cap-kw", type=float, default=d.grid_import_cap_kw)
    g.add_argument("--grid-export-cap-kw", type=float, default=d.grid_export_cap_kw)
    g.add_argument("--peak-charge-yen-kw", type=float, default=d.peak_demand_charge_yen_kw)
    g.add_argument("--degradation-yen-kwh", type=float, default=d.degradation_cost_yen_kwh)
    return p.parse_args()


def load_window(path: Path, start: str | None, hours: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    window = df.iloc[:hours].reset_index(drop=True)
    if window.empty:
        raise SystemExit(f"No data in {path} for start={start!r}")
    if len(window) < hours:
        print(f"Warning: only {len(window)} hours available (requested {hours}).")
    return window


def schedule_frame(window: pd.DataFrame, res: DispatchResult) -> pd.DataFrame:
    T = len(window)
    return pd.DataFrame(
        {
            "timestamp": window["timestamp"],
            "load_kw": window["load_kw"],
            "solar_kw": window["solar_kw"],
            "import_price_yen_kwh": window["import_price_yen_kwh"],
            "grid_import_kw": np.round(res.p_imp_kw, 1),
            "grid_export_kw": np.round(res.p_exp_kw, 1),
            "charge_kw": np.round(res.p_ch_kw, 1),
            "discharge_kw": np.round(res.p_dis_kw, 1),
            "soc_end_kwh": np.round(res.soc_kwh[1 : T + 1], 1),
            "slack_import_kw": np.round(res.slack_imp_kw, 1),
            "slack_soc_kwh": np.round(res.slack_soc_kwh[1 : T + 1], 1),
        }
    )


def main() -> None:
    args = parse_args()
    params = replace(
        BessParams(),
        capacity_kwh=args.capacity_kwh,
        max_charge_kw=args.max_charge_kw,
        max_discharge_kw=args.max_discharge_kw,
        init_soc_kwh=args.init_soc_kwh,
        end_soc_min_kwh=args.end_soc_min_kwh,
        grid_import_cap_kw=args.grid_import_cap_kw,
        grid_export_cap_kw=args.grid_export_cap_kw,
        peak_demand_charge_yen_kw=args.peak_charge_yen_kw,
        degradation_cost_yen_kwh=args.degradation_yen_kwh,
    )

    window = load_window(args.data, args.start, args.hours)
    T = len(window)
    timestamps = window["timestamp"].astype(str).tolist()
    print(f"Horizon: {timestamps[0]} -> {timestamps[-1]} ({T} h) | Battery {params.capacity_kwh:.0f} kWh / "
          f"{params.max_charge_kw:.0f} kW")

    soc_min = grid_cap = None
    if args.alert:
        patch = translate_exception_text(args.alert, capacity_kwh=params.capacity_kwh)
        print("\nAlert overrides:")
        for line in patch.diff_summary():
            print(f"  - {line}")
        if patch.end_soc_min_kwh is not None:
            params = replace(params, end_soc_min_kwh=patch.end_soc_min_kwh)
        soc_min_full, grid_cap = patch.apply(T, 0.10 * params.capacity_kwh, params.grid_import_cap_kw)
        soc_min = soc_min_full[:T]

    strict, recovery = solve_dispatch(
        window,
        params,
        soc_min_kwh=soc_min,
        grid_import_cap_kw=grid_cap,
        auto_recover_infeasible=not args.no_recovery,
    )

    print("\n" + build_operator_brief(strict, recovery, timestamps))

    final = strict if strict.status == "OPTIMAL" else recovery
    if args.out:
        if final is None or final.status != "OPTIMAL":
            print(f"\nNo feasible schedule to write to {args.out}.")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            schedule_frame(window, final).to_csv(args.out, index=False)
            print(f"\nSchedule written to {args.out}")


if __name__ == "__main__":
    main()
