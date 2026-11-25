---
name: bess-model-verifier
description: Verify the BESS dispatch model after changing core_model/formulation.py, core_model/solver_runner.py, core_model/evaluation.py, or the data pipeline. Runs the physics / cross-solver / recovery report and the tests, and diagnoses failures before a schedule or result is reported as correct.
---

# BESS model verifier

Use this skill whenever you change the optimization model, a solver backend, cost accounting, or how
site data is prepared, and before you quote any cost, savings or solve-time number.

## 1. Run the verification report

```bash
python verify.py          # day / week / month on HiGHS, SCIP and the SciPy reference (~15 s)
python verify.py --quick  # while iterating
```

It checks, for every horizon and solver: power balance, SoC dynamics and bounds, no simultaneous
charge/discharge, that all backends prove the **same optimal cost**, and that an infeasible scenario is
recovered with violations located in time. Exit code 1 means something failed. Quote the failing lines.

## 2. Run the tests

```bash
pytest -q
```

## 3. If something fails

- **Backends disagree on cost.** One formulation is wrong. The JijModeling model (`build_problem`) and the
  hand-built SciPy model (`_solve_with_scipy`) are independent implementations of the same equations; check
  each solution against the physics (`verify.physics_failures`) to find which one is wrong. This is how a
  swapped charge/discharge sign in the SciPy power balance was found.
- **Agreement looks "almost" right (e.g. 0.006 %).** Make sure both sides used `SolverOptions(mip_rel_gap=1e-9)`.
  HiGHS stops at a 1e-4 relative gap by default, so a correct solver can return a near-optimal cost.
- **Physics check fails on one backend only.** Usually a variable-layout or sign error in that backend.
- **Strict solve is infeasible.** Do not relax constraints yourself. Read the elastic recovery's
  `infeasibility_summary()`. It gives the exact hours and magnitudes of `s_imp_cap` / `s_soc_min`. Report those,
  then propose relaxations.

## 4. Rules when changing the model

- Add every new constraint or variable to **both** `core_model/formulation.py` and `_solve_with_scipy`, and
  extend `tests/test_formulation.py` with a physics check for it.
- Keep strict and elastic as one model. Slack is controlled by its upper bound (`slack_ub_*`), not by a
  second formulation.
- If savings numbers change, re-run `python benchmarks/run_benchmarks.py` and update the README tables.
