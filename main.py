"""Command-line entry point for the BESS + PV dispatch workbench.

Examples:
    python main.py                                   # first 24h of the dataset
    python main.py --start 2026-04-07 --hours 168    # one week from Apr 7
    python main.py --alert "Typhoon warning, cap grid import at 300 kW"
    python main.py --solver scip --capacity-kwh 3000 --out output/schedule.csv
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from core_model.data_validation import DataValidationError
from core_model.evaluation import format_comparison
from core_model.formulation import BessParams
from core_model.logging_utils import setup_logging
from core_model.solver_runner import SOLVERS
from service.pipeline import DEFAULT_DATA, load_data, run_scenario, select_window


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize hourly battery dispatch for a solar + BESS site.")
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Hourly site CSV (see data/prepare_japan_grid_data.py)"
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start timestamp, e.g. 2026-04-07 or '2026-04-07 06:00'. Default: first row",
    )
    p.add_argument(
        "--hours", type=int, default=24, help="Horizon length in hours (default 24; 168 = week, 720 = month)"
    )
    p.add_argument(
        "--alert", type=str, default=None, help="Operator alert text to translate into reserve / grid-cap overrides"
    )
    p.add_argument(
        "--solver", choices=SOLVERS, default="highs", help="Solver backend (default: highs via JijModeling + OMMX)"
    )
    p.add_argument(
        "--no-recovery", action="store_true", help="Do not run the elastic (slack) re-solve when strict is infeasible"
    )
    p.add_argument(
        "--no-compare", action="store_true", help="Skip the comparison against no-battery / rule-based baselines"
    )
    p.add_argument("--out", type=Path, default=None, help="Write the hourly schedule to this CSV")
    p.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING (default: $LOG_LEVEL or WARNING)")
    p.add_argument("--log-json", action="store_true", help="Emit JSON log lines (default: $LOG_FORMAT=json)")

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


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, json_format=True if args.log_json else None)

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

    window = select_window(load_data(args.data), args.start, args.hours)
    if window.empty:
        print(f"No data in {args.data} for start={args.start!r}", file=sys.stderr)
        return 2
    if len(window) < args.hours:
        print(f"Warning: only {len(window)} hours available (requested {args.hours}).")

    ts = window["timestamp"].astype(str)
    print(
        f"Horizon: {ts.iloc[0]} -> {ts.iloc[-1]} ({len(window)} h) | "
        f"Battery {params.capacity_kwh:.0f} kWh / {params.max_charge_kw:.0f} kW | Solver: {args.solver}"
    )

    try:
        result = run_scenario(
            window,
            params,
            alert=args.alert,
            solver=args.solver,
            auto_recover=not args.no_recovery,
            compare=not args.no_compare,
        )
    except DataValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"Note: {w}")
    if result.patch is not None:
        print("\nAlert overrides:")
        for line in result.patch.diff_summary():
            print(f"  - {line}")

    print("\n" + result.brief)

    if result.comparison is not None:
        print("\nValue vs baselines:")
        print(format_comparison(result.comparison))

    if args.out:
        sched = result.schedule_frame()
        if sched is None:
            print(f"\nNo feasible schedule to write to {args.out}.")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            sched.to_csv(args.out, index=False)
            print(f"\nSchedule written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
