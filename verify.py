"""Model verification report: run after any change to the formulation, solvers or data pipeline.

    python verify.py            # day / week / month, all solvers (~15 s)
    python verify.py --quick    # day / week, HiGHS + SciPy reference only

Checks (the bess-model-verifier checklist, automated):
  1. power balance      |load + ch + exp - solar - dis - imp| < 1e-5 every step
  2. SoC trajectory     soc[0] = init, dynamics hold, reserve floor <= soc <= capacity, end floor met
  3. mode lock          never charging and discharging in the same step
  4. cross-solver       every backend proves the same optimal cost (rel. 1e-6, MIP gap 1e-9)
  5. recovery           an infeasible scenario is recovered, with violations located in time
  6. runtime            solve time per horizon and backend (reported; fails only above --max-seconds)

Exit code 0 = all checks passed, 1 = at least one failed.
"""

import argparse
import sys

import numpy as np

from core_model.formulation import BessParams
from core_model.solver_runner import SOLVERS, DispatchResult, SolverOptions, solve_dispatch
from service.pipeline import load_data

TOL = 1e-5
# Solvers stop at their default MIP gap (HiGHS: 1e-4), so two correct solvers can return
# different near-optimal costs. Verification asks every solver to prove optimality instead.
EXACT = SolverOptions(mip_rel_gap=1e-9)


def physics_failures(window, r: DispatchResult, p: BessParams) -> list[str]:
    fails = []
    bal = window["load_kw"] + r.p_ch_kw + r.p_exp_kw - window["solar_kw"] - r.p_dis_kw - r.p_imp_kw
    if np.abs(bal).max() > TOL:
        fails.append(f"power balance off by {np.abs(bal).max():.2e} kW")
    dyn = r.soc_kwh[1:] - (
        r.soc_kwh[:-1] + p.charge_eff * r.p_ch_kw * p.dt_hours - r.p_dis_kw * p.dt_hours / p.discharge_eff
    )
    if np.abs(dyn).max() > TOL:
        fails.append(f"SoC dynamics off by {np.abs(dyn).max():.2e} kWh")
    if abs(r.soc_kwh[0] - p.init_soc_kwh) > TOL:
        fails.append("initial SoC mismatch")
    if r.soc_kwh[-1] < p.end_soc_min_kwh - TOL:
        fails.append("end SoC below floor")
    floor = p.default_soc_min_frac * p.capacity_kwh
    if r.soc_kwh.min() < floor - TOL or r.soc_kwh.max() > p.capacity_kwh + TOL:
        fails.append("SoC outside [reserve, capacity]")
    simul = np.minimum(r.p_ch_kw, r.p_dis_kw)
    if simul.max() > TOL:
        fails.append(f"simultaneous charge/discharge in {int((simul > TOL).sum())} step(s)")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description="BESS model verification report")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="Fail if any single solve exceeds this")
    args = ap.parse_args()

    df, p = load_data(), BessParams()
    horizons = [24, 168] if args.quick else [24, 168, 720]
    solvers = ["highs", "scipy"] if args.quick else list(SOLVERS)
    failures: list[str] = []

    print(f"{'horizon':>8} {'solver':>6} {'status':>9} {'cost (yen)':>14} {'time s':>7}  checks")
    for T in horizons:
        window = df.iloc[:T].reset_index(drop=True)
        costs = {}
        for s in solvers:
            r, _ = solve_dispatch(window, p, solver=s, options=EXACT)
            fails = physics_failures(window, r, p) if r.usable else ["no usable solution"]
            if r.solve_time_sec > args.max_seconds:
                fails.append(f"slow: {r.solve_time_sec:.1f}s > {args.max_seconds}s")
            costs[s] = r.objective_yen
            failures += [f"T={T} {s}: {f}" for f in fails]
            cost = f"{r.objective_yen:,.0f}" if r.objective_yen is not None else "-"
            print(
                f"{T:>8} {s:>6} {r.status:>9} {cost:>14} {r.solve_time_sec:>7.2f}  {'OK' if not fails else '; '.join(fails)}"
            )
        ref = costs[solvers[0]]
        for s, c in costs.items():
            if ref is None or c is None or abs(c - ref) > 1e-6 * abs(ref):
                failures.append(f"T={T}: {s} cost {c} differs from {solvers[0]} {ref}")

    # Recovery: 75 % reserve + 300 kW cap on a ~700 kW load day cannot be met.
    day = df.iloc[:24].reset_index(drop=True)
    strict, rec = solve_dispatch(day, p, soc_min_kwh=np.full(24, 1500.0), grid_import_cap_kw=np.full(24, 300.0))
    notes = rec.infeasibility_summary(day["timestamp"].astype(str).tolist()) if rec else []
    ok = strict.status == "INFEASIBLE" and rec is not None and rec.usable and notes
    print(
        f"\nrecovery: strict={strict.status}, elastic={rec.status if rec else '-'}, located violations={len(notes)}"
        f"  {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append("infeasible scenario was not recovered with located violations")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S):"))
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
