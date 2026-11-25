---
name: bess-model-verifier
description: Verify BESS and grid dispatch optimization formulations, check physical conservation laws, run horizon benchmarks, and diagnose infeasibility before claiming a schedule is production-ready.
---

# BESS Optimization Model Verifier Skill

Use this skill whenever modifying `core_model/formulation.py`, `core_model/solver_runner.py`, or scenario/data pipelines.

## Verification Checklist
1. **Power Balance Check:** Confirm `|load[t] + p_ch[t] + p_exp[t] - solar[t] - p_dis[t] - p_imp[t]| < 1e-5` for every time step `t`.
2. **State of Charge (SoC) Trajectory Check:** Confirm `soc[0] == soc_init`, `soc[t+1] == soc[t] + eta_ch * p_ch[t] * dt - (1 / eta_dis) * p_dis[t] * dt`, and `0 <= soc[t] <= capacity_kwh`.
3. **No Simultaneous Charge & Discharge:** Verify `min(p_ch[t], p_dis[t]) < 1e-5` across all `t` (binary mode lock working as intended).
4. **Infeasibility & Slack Diagnostics:** When a scenario is infeasible under strict limits, run the elastic slack fallback and report exact timestamps and magnitudes of `s_soc_min` and `s_imp_cap` before proposing constraint relaxations.
5. **Runtime & Scaling Telemetry:** Report wall-clock solve time and status across horizons (`T = 24`, `168`, `720`) so performance regressions are caught early.