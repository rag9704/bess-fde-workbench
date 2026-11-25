---
name: bess-model-verifier
description: Verify BESS and grid dispatch optimization formulations, check physical conservation laws, run horizon benchmarks, and diagnose infeasibility before claiming a schedule is production-ready.
---

# BESS Optimization Model Verifier Skill

Use this skill whenever modifying `core_model/formulation.py`, `core_model/solver_runner.py`, or scenario/data pipelines.

## Run the automated checks first
Checks 1–4 and cross-solver agreement are automated. Run them before anything else and report the result:

```bash
pytest tests/test_formulation.py tests/test_solver_agreement.py tests/test_recovery.py tests/test_evaluation.py -q
```

- `test_solver_agreement.py` solves the same scenarios with JijModeling+OMMX/HiGHS, JijModeling+OMMX/SCIP and the
  hand-built SciPy model. If they disagree, one formulation has a bug: find it before trusting any result.
- When you add a constraint or variable, add it to **both** `core_model/formulation.py` and `_solve_with_scipy` in
  `core_model/solver_runner.py`, and extend the physics tests for it.

## Verification Checklist
1. **Power Balance Check:** Confirm `|load[t] + p_ch[t] + p_exp[t] - solar[t] - p_dis[t] - p_imp[t]| < 1e-5` for every time step `t`.
2. **State of Charge (SoC) Trajectory Check:** Confirm `soc[0] == soc_init`, `soc[t+1] == soc[t] + eta_ch * p_ch[t] * dt - (1 / eta_dis) * p_dis[t] * dt`, and `0 <= soc[t] <= capacity_kwh`.
3. **No Simultaneous Charge & Discharge:** Verify `min(p_ch[t], p_dis[t]) < 1e-5` across all `t` (binary mode lock working as intended).
4. **Infeasibility & Slack Diagnostics:** When a scenario is infeasible under strict limits, run the elastic slack fallback and report exact timestamps and magnitudes of `s_soc_min` and `s_imp_cap` before proposing constraint relaxations.
5. **Runtime & Scaling Telemetry:** Report wall-clock solve time and status across horizons (`T = 24`, `168`, `720`) so performance regressions are caught early.