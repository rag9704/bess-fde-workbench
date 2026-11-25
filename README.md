# BESS FDE Workbench

Cost-optimal hourly dispatch for a **commercial solar + battery (BESS) site in Japan**, built on real
TEPCO area demand/solar data and JEPX Tokyo spot prices.

- **Model:** a mixed-integer linear program written once in **JijModeling**, compiled to an **OMMX** instance and
  solved with **HiGHS** or **SCIP**. An independent SciPy model is kept as a cross-check.
- **Measured value:** every schedule is compared with *no battery* and with a *rule-based* operating policy
  using the same cost function.
- **AI harness:** operator alerts in English or Japanese ("台風接近、受電を300kWに制限") are read by **Claude**
  into a typed schema. Deterministic guardrails then clamp every value, and overrides can only *tighten*
  limits. A 30-case bilingual eval measures it. When the limits can't all be met, the optimizer reports
  *which* limit breaks, *when*, and *by how much*, and Claude can explain the trade-off to the operator.
- **Benchmarked:** scaling to 30-min resolution, a quality-vs-time study, and an exact **LP-first** strategy
  that returns the proven MILP optimum about 2.5× faster on real data.
- **Production-ready:** CLI, FastAPI service, Docker image, input checks, structured logging, 70+ tests, a
  model-verification report, Claude Code skills for the team, and GitHub Actions CI.

**Results on April 2026 data** (2,000 kWh / 500 kW battery, 600 kW PV, 1,000 kW peak load):

| Horizon | No battery | Rule-based | **Optimized** | Optimized saving | Peak import |
|---|---|---|---|---|---|
| 1 day | ¥2,093,372 | ¥2,069,871 (−1.1%) | **¥1,860,918** | **−11.1%** | 986 → 827 kW |
| 1 week | ¥5,408,155 | ¥5,279,127 (−2.4%) | **¥5,044,001** | **−6.7%** | 986 → 799 kW |
| 1 month | ¥16,031,682 | ¥15,334,758 (−4.3%) | **¥15,033,602** | **−6.2%** | 986 → 813 kW |

Most of the value comes from **peak shaving**: the rule-based policy never lowers the demand-charge peak,
while the optimizer cuts it by about 17–19%.

---

## 1. The problem

A site has electricity **load**, **solar generation**, a **battery** and a **grid connection**. For every
hour, decide how much to **import / export** and when to **charge / discharge**, to minimize:

```
  Σ_t import_price[t] · import[t] · dt          energy bought
− Σ_t export_price[t] · export[t] · dt          energy sold
+ Σ_t wear_cost · (charge[t] + discharge[t]) · dt   battery degradation
+ demand_charge · max_t import[t]               peak (kW) demand charge
```

subject to:

| Constraint | Meaning |
|---|---|
| `power_balance` | `load + charge + export = solar + discharge + import` every hour |
| `soc_dynamics` | `soc[t+1] = soc[t] + η_ch·charge·dt − discharge·dt/η_dis` |
| `soc_init_eq`, `soc_end_floor` | start at `init_soc_kwh`, finish at or above `end_soc_min_kwh` |
| `soc_min_reserve`, `soc_upper_bound` | reserve floor (default 10%) and ceiling at every state |
| `charge_mode_limit`, `discharge_mode_limit` | a binary `z_ch[t]` prevents charging and discharging in the same hour |
| `grid_import_cap`, export bound | grid connection limits |
| `peak_import_def` | `p_peak ≥ import[t]` (linearizes the max for the demand charge) |

### Strict and elastic in one model

The slack variables `s_soc_min` and `s_imp_cap` are always in the model, and their upper bound is a parameter:

- **strict:** `slack_ub = 0`, so every limit is hard.
- **elastic:** `slack_ub` is large and slack costs ¥10,000/kWh, so the solver breaks limits only when it has to.
  The non-zero slacks show exactly where the scenario is infeasible.

Because the horizon length `T` is also a parameter, **the model is built once** (`build_problem()`, cached) and
reused for every site, horizon and scenario. Only the instance data changes.

---

## 2. Project structure

```
bess-fde-workbench/
├── main.py                        # CLI
├── core_model/
│   ├── formulation.py             # BessParams + JijModeling model + instance data builder
│   ├── solver_runner.py           # backends (OMMX/HiGHS, OMMX/SCIP, SciPy reference), strict → elastic fallback
│   ├── evaluation.py              # cost breakdown, no-battery & rule-based baselines, comparison table
│   ├── data_validation.py         # input checks (columns, gaps, duplicates, NaN, negatives)
│   └── logging_utils.py           # text or JSON structured logs
├── ai_harness/
│   ├── llm_translator.py          # alert → Claude structured extraction → guardrails → overrides
│   ├── exception_translator.py    # regex alert parser (offline baseline + fallback)
│   ├── llm_explainer.py           # Claude-written explanation of limit conflicts
│   ├── diagnostic_agent.py        # results → operator summary
│   └── evals/                     # 30 labelled EN/JA alerts + scorer (run_alert_eval.py)
├── service/
│   ├── pipeline.py                # validate → translate alert → solve → compare (shared by CLI and API)
│   └── api.py                     # FastAPI app: GET /health, POST /optimize
├── data/
│   ├── prepare_japan_grid_data.py # raw TEPCO + JEPX → hourly site CSV
│   ├── april_2026_hourly.csv      # ready-to-use dataset (720 h)
│   ├── april_2026_30min.csv       # same month at 30-min resolution (1,440 steps; benchmarks)
│   └── raw/                       # TEPCO eria_jukyu + JEPX spot_summary
├── benchmarks/                    # run_benchmarks.py → RESULTS.md (scaling, gap/time, LP relaxation)
├── verify.py                      # model verification report (physics, cross-solver, recovery, timing)
├── .claude/skills/                # Claude Code skills: bess-model-verifier, alert-translator-eval
├── tests/                         # pytest suite
├── Dockerfile, .github/workflows/ci.yml, pyproject.toml, requirements*.txt
```

```
 CSV / API request
        │
        ▼
 data_validation ──► alert translator: rules | Claude + guardrails (optional alert)
        │                       │ overrides
        ▼                       ▼
 make_instance_data ──► JijModeling problem.eval() ──► OMMX Instance ──► HiGHS / SCIP
                                                                            │
         strict infeasible? ──► same model, elastic slack bounds ◄──────────┘
        │
        ▼
 evaluation (vs baselines) ──► diagnostic_agent brief ──► CLI output / JSON response
```

---

## 3. Setup

Python 3.12+ (tested on 3.12–3.14).

```bash
cd bess-fde-workbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # or requirements.txt for runtime only
```

---

## 4. Command line

```bash
python main.py                                        # first 24 h
python main.py --start 2026-04-07 --hours 168         # one week
python main.py --hours 720                            # whole month (~1.5 s)
python main.py --solver scip                          # highs (default) | scip | scipy
python main.py --capacity-kwh 3000 --max-charge-kw 750 --max-discharge-kw 750
python main.py --hours 168 --out output/week.csv      # save the hourly schedule
python main.py --log-level INFO --log-json            # structured logs to stderr
python main.py --hours 720 --lp-first                 # exact LP-first speed-up (same optimum)
python main.py --hours 720 --time-limit 2 --mip-gap 0.01   # trade optimality for time
```

Example output:

```
Horizon: 2026-04-01 00:00:00 -> 2026-04-01 23:00:00 (24 h) | Battery 2000 kWh / 500 kW | Solver: highs

Status: OPTIMAL (JijModeling + OMMX / HiGHS, 0.297s) | Objective: ¥1,854,251 | Peak Import: 827.2 kW
Energy: Import 19,473 kWh, Export 0 kWh, Charge 1,100 kWh, Discharge 1,160 kWh

Value vs baselines:
Policy                                 Cost (adj.)   Peak kW    Demand chg       Savings       %
-----------------------------------------------------------------------------------------------
No battery                              ¥2,093,372     986.5    ¥1,381,100            ¥0    0.0%
Rule-based (cheap/expensive hours)      ¥2,069,871     986.5    ¥1,381,100       ¥23,500    1.1%
Optimized (MILP)                        ¥1,860,918     827.2    ¥1,158,011      ¥232,454   11.1%
```

### Operator alerts

```bash
python main.py --alert "Typhoon warning, cap grid import at 300 kW"
```

```
Alert overrides (read by: rules):
  - SoC reserve floor >= 1500.0 kWh for hour indices [0..23]
  - Grid import cap <= 300.0 kW for hour indices [0..23]
  - Applied 75% storm/outage reserve heuristic

Status: STRICT INFEASIBLE (0.303s) — hard constraints conflicted under current inputs.
Elastic Recovery: OPTIMAL (0.254s) | Operational cost (excl. penalties): ¥2,383,220 | Peak Import: 1192.0 kW
Detected constraint conflicts (top items):
  - Grid import cap exceeded by 892.0 kW at 2026-04-01 00:00:00
  - Grid import cap exceeded by 394.9 kW at 2026-04-01 01:00:00
  ...
```

With an alert the baseline comparison is skipped, because the baselines don't model alert overrides and
the comparison wouldn't be like-for-like.

**Who reads the alert** (`--translator`):

| Mode | What happens | Needs |
|---|---|---|
| `rules` (default) | regex/keyword parser. Free, offline, deterministic | nothing |
| `llm` | Claude extracts the request into a typed schema; code applies the guardrails | `ANTHROPIC_API_KEY` |
| `auto` | Claude, falling back to `rules` on any API error, refusal or missing key | optional key |

```bash
export ANTHROPIC_API_KEY=...
python main.py --translator llm --alert "停電リスクあり。14時から18時まで予備率70%を確保"
python main.py --translator auto --explain --alert "Typhoon warning, cap grid import at 300 kW"
```

`--explain` adds a short Claude-written note when limits conflict: why they conflict, and the decision the
operator has to make.

What the **rule parser** understands:

| You write | Effect |
|---|---|
| `reserve 80%`, `soc 60%` | SoC floor = % of capacity |
| `reserve 1200 kWh` | SoC floor in kWh |
| `typhoon`, `storm`, `outage`, `台風`, `警報`, `停電` (no number) | SoC floor = 75% of capacity |
| `cap grid import at 300 kW`, `transformer limit 500kw`, `受電 600kW` | import cap |
| `from 14 to 20`, `at 13:00 to 17:00`, `hours14-20` | hour window (index into the horizon; default 0–23) |

It scores 16/30 on the eval set (section 8). Two known gaps are tracked as `xfail` tests: `90% reserve`
(number before the keyword) and `hours 14-20` (with a space).

### All options

| Option | Default | |
|---|---|---|
| `--data` | `data/april_2026_hourly.csv` | hourly input CSV |
| `--start` / `--hours` | first row / 24 | horizon |
| `--alert` | – | operator alert text |
| `--translator` / `--explain` | `rules` / off | how to read `--alert`; Claude explanation of conflicts |
| `--solver` | `highs` | `highs`, `scip`, `scipy` |
| `--lp-first` | off | exact speed-up: accept the LP relaxation when it is MILP-feasible |
| `--mip-gap` / `--time-limit` | solver default / none | stop early (status `FEASIBLE` if the time limit hits) |
| `--no-recovery` / `--no-compare` | off | skip elastic re-solve / baselines |
| `--out` | – | write schedule CSV |
| `--log-level` / `--log-json` | `WARNING` / text | also via `LOG_LEVEL`, `LOG_FORMAT=json` |
| `--capacity-kwh` | 2000 | battery energy |
| `--max-charge-kw` / `--max-discharge-kw` | 500 / 500 | battery power |
| `--init-soc-kwh` / `--end-soc-min-kwh` | 1000 / 800 | start SoC / minimum end SoC |
| `--grid-import-cap-kw` / `--grid-export-cap-kw` | 1200 / 400 | grid limits |
| `--peak-charge-yen-kw` | 1400 | demand charge |
| `--degradation-yen-kwh` | 2.0 | wear cost per kWh throughput |

---

## 5. HTTP API

```bash
uvicorn service.api:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

```bash
curl -s -X POST localhost:8000/optimize -H 'content-type: application/json' -d '{
  "start": "2026-04-07", "hours": 168,
  "battery": {"capacity_kwh": 3000, "max_charge_kw": 750},
  "solver": "highs",
  "include_schedule": false
}'
```

| Field | Description |
|---|---|
| `start`, `hours` (1–744) | horizon within the data |
| `series` | optional: your own rows `{timestamp, load_kw, solar_kw, import_price_yen_kwh, export_price_yen_kwh}` |
| `battery` | any `BessParams` field; omitted fields use defaults |
| `alert` | operator alert text |
| `translator`, `explain` | `rules` \| `llm` \| `auto`; Claude explanation of conflicts |
| `solver`, `lp_first`, `mip_rel_gap`, `time_limit_sec` | solver backend and quality/time knobs |
| `auto_recover`, `compare`, `include_schedule` | as in the CLI |

The response contains:

- `status`: `OPTIMAL`, `FEASIBLE` (time limit hit, usable schedule), `RECOVERED`, `INFEASIBLE` or `NO_SOLUTION`
- results: `objective_yen`, `mip_gap`, `peak_import_kw`, `schedule` (hourly rows)
- explanation: `brief`, `overrides`, `translator_source`, `violations`, `explanation`
- `comparison`: the cost breakdown per policy

With `translator: "llm"` and Claude unavailable, the API returns **503**. Use `auto` to fall back instead.

If the data fails the input checks, the API returns **422 with every issue found**, not just the first:

```json
{"detail": {"message": "Input data failed validation",
            "issues": ["1 gap(s)/irregular step(s) (expected 1.0 h), first at 2026-01-01 03:00:00",
                       "Column 'load_kw' has 1 negative value(s)"]}}
```

Every response carries an `x-request-id` header. With `LOG_LEVEL=INFO` each request and solve is logged
with its duration, solver, horizon, status and objective.

### Docker

```bash
docker build -t bess-optimizer .
docker run -p 8000:8000 bess-optimizer       # JSON logs, honours $PORT (Cloud Run compatible)
```

---

## 6. Using it from Python

```python
from core_model.formulation import BessParams
from service.pipeline import load_data, select_window, run_scenario

window = select_window(load_data(), start="2026-04-07", hours=168)
result = run_scenario(window, BessParams(capacity_kwh=2500), solver="highs")

print(result.brief)
print(result.comparison[["adjusted_cost_yen", "peak_import_kw", "savings_pct"]])
schedule = result.schedule_frame()  # pandas DataFrame, one row per hour
```

Lower level:

```python
from core_model.formulation import build_problem, make_instance_data
from core_model.solver_runner import solve_dispatch

problem = build_problem()  # jm.Problem (renders as LaTeX in Jupyter)
instance = problem.eval(make_instance_data(window, BessParams()))  # ommx.v1.Instance
strict, recovery = solve_dispatch(window, BessParams(), solver="scip")
```

---

## 7. How savings are measured

All policies are scored by `core_model/evaluation.py` with the optimizer's cost function (slack penalties excluded):

- **No battery:** solar and grid only. Surplus PV is exported up to the export cap, then curtailed.
- **Rule-based:** what a site typically runs without an optimizer. Per day it charges in the 6 cheapest hours
  (without pushing import above the existing peak), discharges to cover load in the 6 most expensive hours,
  and always stores surplus solar.
- **Optimized:** the MILP schedule.

**Stored-energy adjustment:** a policy that ends the horizon with less energy than it started with has used
"free" stored energy. The change in SoC is valued at the horizon's average import price × discharge efficiency
and added to the cost, so policies are compared like-for-like.

---

## 8. AI harness: design and evaluation

### Claude extracts, code decides

`ai_harness/llm_translator.py` calls Claude Opus 5.5 (`claude-opus-5-5`, low effort) with **structured outputs**:
`client.beta.messages.parse(output_format=AlertExtraction)`. The schema has explicit reserves, import caps, end-of-horizon
SoC, a storm/outage flag, and a `not_applied` list for requests that have no matching control. Server-side
refusal fallback (`fallbacks="default"`) is on.

Everything that touches the physical schedule is decided by `extraction_to_patch`, not by the model:

| Rule | Why |
|---|---|
| Hours are clamped into the horizon, reversed windows are swapped, out-of-horizon requests are dropped with a note | a misread time can't break the instance |
| `0 ≤ reserve ≤ capacity`, `0–100 %`, caps `≥ 0` | a reserve above capacity would make every schedule infeasible |
| The 75% storm reserve is site policy applied in code; the model only flags the risk | policy changes are code reviews, not prompt edits |
| Overrides can only **tighten** limits (max for floors, min for caps, never lower the end-SoC floor) | a bad extraction can make the schedule more conservative, never less safe |
| Requests with no matching control are shown as "Not applied: …" | nothing is dropped silently |
| `auto` mode falls back to rules on API errors, refusals, truncation or a missing key | the optimizer never blocks on the LLM |

### Evaluation

`ai_harness/evals/alert_cases.jsonl` holds 30 labelled alerts in English and Japanese. They cover clock windows,
MW→kW, end-of-horizon requests, two windows in one alert, clamping, storm policy, out-of-scope requests, and
"no action needed". A case passes only if every override matches.

```bash
python -m ai_harness.evals.run_alert_eval --mode rules     # offline baseline
python -m ai_harness.evals.run_alert_eval --mode llm --out ai_harness/evals/results_llm.json   # ~30 Claude calls
```

| Translator | Accuracy | Notes |
|---|---|---|
| Rule parser | **16/30 (53%)** | fails on clock windows, Japanese windows, MW, end-of-horizon SoC, number-before-keyword |
| Claude Opus 5.5 + guardrails | *run `--mode llm` with your key* | the report prints per-field accuracy, every failure and the token cost |

All Claude-path tests run offline with a fake client (`tests/test_llm_harness.py`): request shape, guardrails,
every fallback path, only-tighten behaviour, and a check that the eval labels agree with the code policy.

### Harness for the development team

- **`verify.py`**: one command that checks power balance, SoC dynamics, the mode lock, that every backend proves
  the same optimal cost (MIP gap 1e-9), and infeasibility recovery, with timings. Exit code 1 on any failure.
- **`.claude/skills/bess-model-verifier`**: tells Claude Code to run `verify.py` and the tests after any model
  change, and how to diagnose each kind of failure.
- **`.claude/skills/alert-translator-eval`**: the design rules above, plus the measure-before-and-after
  workflow for changing the prompt, schema, rules or eval cases.

---

## 9. Verification, tests and benchmarks

```bash
pytest                      # ~15 s, offline
python verify.py            # model verification report
ruff check . && ruff format --check .
python benchmarks/run_benchmarks.py    # ~2 min → benchmarks/RESULTS.md
```

| Test file | What it checks |
|---|---|
| `test_formulation.py` | power balance, SoC dynamics and bounds, no simultaneous charge/discharge, power limits, zero slack when strict |
| `test_solver_agreement.py` | HiGHS, SCIP and the SciPy reference prove the **same optimal cost** (normal, week-long, infeasible) |
| `test_solver_options.py` | LP-first is exact (and falls back when the LP cheats), time limits, gaps, LP lower bound |
| `test_recovery.py` | infeasible scenario → elastic recovery with located violations |
| `test_evaluation.py` | baselines obey physics, the rule never raises the peak, the optimizer beats both, cost breakdown = solver objective |
| `test_data_validation.py` | missing columns, gaps, duplicates, NaN, negatives (all reported at once) |
| `test_exception_translator.py`, `test_llm_harness.py` | rule parser, Claude path with a fake client, eval set integrity |
| `test_api.py`, `test_cli.py` | end-to-end through FastAPI and the CLI |

**Bugs the verification caught:**

- **Swapped signs in the original SciPy model.** When the JijModeling model was introduced, its results
  disagreed with the hand-built SciPy model. Checking each solution against the physics showed the SciPy power
  balance had the charge and discharge signs swapped, and earlier results from it were wrong.
- **A false "disagreement" from the default MIP gap.** `verify.py` later flagged a 0.006% difference at one
  month. HiGHS stops at its default 1e-4 MIP gap, so verification now asks every solver to *prove* optimality.

CI (`.github/workflows/ci.yml`) runs these jobs:

1. Lint.
2. On Python 3.12, 3.13 and 3.14: the tests, `verify.py --quick`, the rule-parser eval and a CLI smoke test.
3. Docker: build the image and call `/optimize` on the running container.

### Benchmark findings

Full tables are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) (April 2026, Apple Silicon laptop):

- **Scaling.** A month at 30-min resolution (1,440 steps and binaries) solves in about 1.2–1.6 s on every
  backend. About 0.5 s of the OMMX path is fixed overhead (JijModeling compile ≈ 0.25–0.3 s, HiGHS model
  build ≈ 0.23 s), which dominates short horizons.
- **Quality vs time.** Relaxing the MIP gap or setting tight time limits saves under a second but costs
  0.03–1.1% (up to ¥168k/month). These instances solve quickly enough that the default settings are the
  right choice.
- **Are the binaries needed?** On the real Tokyo data the LP relaxation has the *same* optimum as the MILP
  (integrality gap 0, no step charges and discharges at once). Round-trip losses and wear cost make burning
  energy unprofitable, and even at negative prices the MILP can burn energy almost as well by alternating
  charge and discharge across steps. The LP only "cheats" in a stress case (negative prices *and* zero
  wear cost: 9 simultaneous steps, 0.001% gap).
- **LP-first (exact).** Solve the LP. If its schedule never charges and discharges at once, it is MILP-feasible
  and therefore MILP-optimal. Otherwise solve the MILP. On real data this returns the proven optimum
  **2.4–2.5× faster**, and in the stress case it correctly falls back.
- **Upstream performance bug found.** `ommx-highs-adapter`'s `decode()` reads `solution.col_value[i]` in a loop.
  highspy copies the whole vector on each access, so decoding is O(n²): 1.7 s for 11.5k variables, 13× the LP
  solve. The workbench reads the vector once (0.04 s). Worth reporting to the OMMX project.

---

## 10. Rebuilding the dataset

```bash
python data/prepare_japan_grid_data.py
python data/prepare_japan_grid_data.py --start-date 2026-04-01 --days 7 --out-csv data/week1.csv
python data/prepare_japan_grid_data.py --freq 30min --out-csv data/april_2026_30min.csv   # raw data is 30-min
```

- **Load:** TEPCO area demand, hourly, scaled so the month's peak = 1,000 kW.
- **Solar:** TEPCO area PV output, scaled so the month's peak = 600 kW.
- **Import price:** JEPX Tokyo area price (30-min slots → hourly) + wheeling fee (default ¥8.5/kWh).
- **Export price:** `min(0.7 × Tokyo price, ¥10/kWh)`.

To use your own site, pass a CSV with `timestamp, load_kw, solar_kw, import_price_yen_kwh, export_price_yen_kwh`
via `--data`, or send rows as `series` to the API.

---

## 11. Known limitations / next steps

- **Perfect foresight.** Load, solar and prices are known for the whole horizon. Next step: forecasts plus
  rolling-horizon re-optimization.
- **Demand charge.** It is applied to the peak within the horizon. Japanese 基本料金 is based on the highest
  **30-minute** demand over 12 months, so the next step is carrying the running peak between solves. The model
  already runs at 30-min resolution (`BessParams(dt_hours=0.5)`), but the CLI and API currently assume hourly data.
- **LLM eval not yet run here.** The Claude path is tested offline with a fake client. Its real accuracy needs
  `run_alert_eval --mode llm` with an API key.
- **Fixed OMMX overhead (~0.5 s per solve).** This matters for many short solves (e.g. a fleet of sites); caching
  the compiled instance and updating only the data would remove most of it.
- **Baselines with alerts.** The baselines don't model alert overrides, so the comparison is skipped when an
  alert is applied.
