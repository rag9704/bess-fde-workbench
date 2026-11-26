"""Reproduce the slow decode as it happened in this project: the JijModeling BESS model, one month at 30 min.

    python reports/ommx-highs-decode/repro_bess_model.py

Builds the OMMX instance exactly as core_model does, solves it with OMMXHighsAdapter, then times:
  1. adapter.decode(model)          - the public API (decode_to_state + instance.evaluate + duals)
  2. adapter.decode_to_state(model) - the part that is O(n^2)
  3. one copy of col_value          - the proposed fix
"""

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ommx.v1 import State  # noqa: E402
from ommx_highs_adapter import OMMXHighsAdapter  # noqa: E402

from core_model.formulation import BessParams, build_problem, make_instance_data  # noqa: E402
from service.pipeline import load_data  # noqa: E402


def main() -> None:
    df = load_data(ROOT / "data" / "april_2026_30min.csv")
    params = BessParams(dt_hours=0.5)

    t0 = time.perf_counter()
    instance = build_problem().eval(make_instance_data(df, params))
    t_compile = time.perf_counter() - t0

    adapter = OMMXHighsAdapter(instance)
    model = adapter.solver_input
    model.setOptionValue("output_flag", False)
    t0 = time.perf_counter()
    model.run()
    t_solve = time.perf_counter() - t0

    t0 = time.perf_counter()
    solution = adapter.decode(model)
    t_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = adapter.decode_to_state(model)
    t_state = time.perf_counter() - t0

    t0 = time.perf_counter()
    col_value = model.getSolution().col_value
    fixed = State(entries={v.id: col_value[i] for i, v in enumerate(instance.used_decision_variables)})
    t_fixed = time.perf_counter() - t0

    assert fixed.entries == state.entries, "fixed decode must return identical values"

    n = len(instance.used_decision_variables)
    print(f"Model: {len(df)} steps at 30 min, {n:,} decision variables, {model.getNumRow():,} rows")
    print(f"  JijModeling eval -> OMMX instance   {t_compile:7.3f} s")
    print(f"  HiGHS run (MILP)                    {t_solve:7.3f} s   objective {solution.objective:,.0f}")
    print(f"  adapter.decode(model)               {t_decode:7.3f} s")
    print(f"    of which decode_to_state          {t_state:7.3f} s")
    print(f"  single copy of col_value (fix)      {t_fixed:7.4f} s   -> {t_state / t_fixed:,.0f}x faster, same values")


if __name__ == "__main__":
    main()
