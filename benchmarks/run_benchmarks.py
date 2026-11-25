"""Solver benchmarks: scaling, quality-vs-time trade-off, and whether the binaries are needed.

    python benchmarks/run_benchmarks.py              # full run, median of 3 repeats (~2-3 min)
    python benchmarks/run_benchmarks.py --repeats 1  # quicker

Writes benchmarks/RESULTS.md (tables + environment) and benchmarks/results/*.csv.

Experiments
  A. Scaling: horizon (day / week / month) x resolution (1 h / 30 min) x solver.
  B. Trade-off: on the largest instance, how much cost is given up for speed via MIP gap / time limits.
  C. Are the binaries needed? Compare the MILP with its LP relaxation (mode-lock binaries in [0, 1]),
     on the real data and on a stress scenario with negative prices.
"""

import argparse
from dataclasses import replace
import datetime as dt
from importlib.metadata import version
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core_model.formulation import BessParams  # noqa: E402
from core_model.solver_runner import SolverOptions, solve_dispatch  # noqa: E402
from service.pipeline import load_data  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
EXACT = SolverOptions(mip_rel_gap=1e-9)
SIMUL_TOL = 1e-4  # kW: a step "charges and discharges at once" above this


def datasets() -> dict[str, tuple[pd.DataFrame, BessParams]]:
    return {
        "1h": (load_data(ROOT / "data" / "april_2026_hourly.csv"), BessParams()),
        "30min": (load_data(ROOT / "data" / "april_2026_30min.csv"), BessParams(dt_hours=0.5)),
    }


def timed_solve(window, params, solver, options, repeats):
    """Median wall-clock over repeats; returns the last result plus timing stats."""
    times, result = [], None
    for _ in range(repeats):
        result, _ = solve_dispatch(window, params, solver=solver, options=options, auto_recover_infeasible=False)
        times.append(result.solve_time_sec)
    return result, statistics.median(times), result.timings.get("compile_sec", 0.0)


def simultaneous_steps(r) -> int:
    return int((np.minimum(r.p_ch_kw, r.p_dis_kw) > SIMUL_TOL).sum())


# --------------------------------------------------------------------------- A. scaling


def exp_scaling(data, repeats) -> pd.DataFrame:
    rows = []
    for res, (df, params) in data.items():
        steps_per_day = int(round(24 / params.dt_hours))
        for label, days in [("day", 1), ("week", 7), ("month", 30)]:
            window = df.iloc[: days * steps_per_day].reset_index(drop=True)
            for solver in ("highs", "scip", "scipy"):
                r, t, compile_t = timed_solve(window, params, solver, None, repeats)
                rows.append(
                    {
                        "horizon": label,
                        "resolution": res,
                        "steps": len(window),
                        "solver": solver,
                        "status": r.status,
                        "time_sec": round(t, 3),
                        "compile_sec": round(compile_t, 3),
                        "cost_yen": round(r.objective_yen or float("nan")),
                        "mip_gap": r.mip_gap,
                    }
                )
                print(f"  A {label:5} {res:5} {solver:5} {r.status:8} {t:6.2f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- B. quality vs time


def exp_tradeoff(data, repeats) -> pd.DataFrame:
    df, params = data["30min"]  # largest instance: 1,440 steps
    rows = []
    for solver in ("highs", "scip"):
        best, t_exact, _ = timed_solve(df, params, solver, EXACT, repeats)
        opt = best.objective_yen
        settings = [
            ("proven optimal (gap 1e-9)", EXACT),
            ("default", SolverOptions()),
            ("gap 0.1%", SolverOptions(mip_rel_gap=1e-3)),
            ("gap 1%", SolverOptions(mip_rel_gap=1e-2)),
            ("gap 5%", SolverOptions(mip_rel_gap=5e-2)),
            ("time limit 0.5 s", SolverOptions(time_limit_sec=0.5)),
            ("time limit 1 s", SolverOptions(time_limit_sec=1.0)),
        ]
        for name, opts in settings:
            r, t, compile_t = (best, t_exact, 0.0) if opts is EXACT else timed_solve(df, params, solver, opts, repeats)
            cost = r.objective_yen
            rows.append(
                {
                    "solver": solver,
                    "setting": name,
                    "status": r.status,
                    "time_sec": round(t, 3),
                    "cost_yen": round(cost) if cost is not None else None,
                    "cost_above_optimal_yen": round(cost - opt) if cost is not None else None,
                    "cost_above_optimal_pct": round(100 * (cost - opt) / opt, 4) if cost is not None else None,
                    "reported_gap": r.mip_gap,
                }
            )
            print(f"  B {solver:5} {name:26} {r.status:11} {t:6.2f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- C. are binaries needed?


def negative_price_scenario(df: pd.DataFrame, price: float = -5.0) -> pd.DataFrame:
    """Midday import price forced negative on sunny hours (as in some EU/AU markets).

    JEPX itself has a ¥0.01/kWh floor, so this is a stress test of the formulation,
    not a Japanese market scenario.
    """
    out = df.copy()
    hour = pd.to_datetime(out["timestamp"]).dt.hour
    out.loc[(hour >= 10) & (hour < 14), "import_price_yen_kwh"] = price
    return out


def exp_relaxation(data, repeats) -> pd.DataFrame:
    rows = []
    df_h, p_h = data["1h"]
    df_30, p_30 = data["30min"]
    scenarios = [
        ("Tokyo Apr 2026, 1 h", df_h, p_h),
        ("Tokyo Apr 2026, 30 min", df_30, p_30),
        ("negative midday prices, 1 h", negative_price_scenario(df_h), p_h),
        (
            "negative prices, zero wear cost, 1 h",
            negative_price_scenario(df_h),
            replace(p_h, degradation_cost_yen_kwh=0.0),
        ),
    ]
    for name, df, params in scenarios:
        milp_r, t_milp, _ = timed_solve(df, params, "highs", EXACT, repeats)
        lp_r, t_lp, _ = timed_solve(df, params, "highs", SolverOptions(relax_integrality=True), repeats)
        lpf_r, t_lpf, _ = timed_solve(df, params, "highs", SolverOptions(lp_first=True), repeats)
        assert abs(lpf_r.objective_yen - milp_r.objective_yen) <= 1e-6 * abs(milp_r.objective_yen), name
        rows.append(
            {
                "scenario": name,
                "milp_cost_yen": round(milp_r.objective_yen),
                "lp_cost_yen": round(lp_r.objective_yen),
                "integrality_gap_pct": round(
                    100 * (milp_r.objective_yen - lp_r.objective_yen) / milp_r.objective_yen, 4
                ),
                "lp_simultaneous_steps": simultaneous_steps(lp_r),
                "milp_time_sec": round(t_milp, 3),
                "lp_time_sec": round(t_lp, 3),
                "lp_first_time_sec": round(t_lpf, 3),
                "lp_first_outcome": "LP accepted" if "MILP-feasible" in lpf_r.backend else "fell back to MILP",
            }
        )
        print(
            f"  C {name:38} MILP {t_milp:5.2f}s  LP {t_lp:5.2f}s  LP-first {t_lpf:5.2f}s  simul={simultaneous_steps(lp_r)}"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def env_info() -> str:
    pkgs = ["jijmodeling", "ommx", "ommx-highs-adapter", "ommx-pyscipopt-adapter", "highspy", "pyscipopt", "scipy"]
    versions = ", ".join(f"{p} {version(p)}" for p in pkgs)
    return (
        f"Run {dt.datetime.now():%Y-%m-%d %H:%M} on {platform.platform()} ({platform.machine()}), "
        f"Python {platform.python_version()}.\n\n{versions}."
    )


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.2g}" if 0 < abs(v) < 1e-3 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per measurement (median reported)")
    args = ap.parse_args()

    data = datasets()
    print("A. scaling")
    a = exp_scaling(data, args.repeats)
    print("B. quality vs time")
    b = exp_tradeoff(data, args.repeats)
    print("C. LP relaxation")
    c = exp_relaxation(data, args.repeats)

    (OUT_DIR / "results").mkdir(exist_ok=True)
    a.to_csv(OUT_DIR / "results" / "scaling.csv", index=False)
    b.to_csv(OUT_DIR / "results" / "tradeoff.csv", index=False)
    c.to_csv(OUT_DIR / "results" / "relaxation.csv", index=False)

    report = f"""# Benchmark results

{env_info()} Times are wall-clock medians of {args.repeats} run(s), including JijModeling compile time
for the OMMX backends. Generated by `python benchmarks/run_benchmarks.py`.

## A. Scaling (default solver settings)

{to_md(a)}

## B. Solution quality vs solve time (month at 30-min resolution, 1,440 steps)

`cost_above_optimal` is measured against the proven optimum, not the solver's own bound.

{to_md(b)}

## C. Are the binary variables needed?

The binary `z_ch[t]` stops the battery charging and discharging in the same step. Its LP relaxation lets
`z_ch` take any value in [0, 1]. If the LP optimum never charges and discharges at once, it is already a
valid schedule and therefore MILP-optimal (LP cost <= MILP cost). `lp_first` uses exactly this: solve the LP,
accept it with that certificate, otherwise fall back to the MILP. The script asserts it always returns the
MILP optimum.

{to_md(c)}
"""
    (OUT_DIR / "RESULTS.md").write_text(report, encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
