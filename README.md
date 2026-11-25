# BESS FDE Workbench

An hourly **battery energy storage (BESS) dispatch optimizer** for a commercial
behind-the-meter site with rooftop solar, built on real Japanese grid data
(TEPCO area demand/solar + JEPX Tokyo spot prices). It also includes a small
"operator harness" that turns plain-text alerts (e.g. *"typhoon warning, cap grid
import at 300 kW"*) into model constraints and explains infeasible results in
plain language.

---

## What problem it solves

A site has electricity **load**, **solar generation**, a **battery** and a
**grid connection**. Every hour, someone has to decide:

- how much to import from / export to the grid,
- when to charge or discharge the battery,

so that the total cost is as low as possible, while respecting physical and
contractual limits.

The cost being minimized over the horizon is:

```
  Σ_t  import_price[t] · grid_import[t]          (energy bought)
- Σ_t  export_price[t] · grid_export[t]          (energy sold)
+ Σ_t  degradation_cost · (charge[t] + discharge[t])   (battery wear)
+ peak_demand_charge · max_t grid_import[t]      (demand / kW charge)
```

Subject to:

| Constraint | Meaning |
|---|---|
| Power balance | `load + charge + export = solar + discharge + import` every hour |
| SoC dynamics | `soc[t+1] = soc[t] + η_ch·charge − discharge/η_dis` |
| Initial / terminal SoC | starts at `init_soc_kwh`, ends ≥ `end_soc_min_kwh` |
| SoC bounds | `soc_min[t] ≤ soc[t] ≤ capacity` (default floor = 10% of capacity) |
| Power limits | charge ≤ max_charge_kw, discharge ≤ max_discharge_kw, export ≤ export cap, import ≤ import cap |
| No simultaneous charge + discharge | enforced with a binary variable per hour (makes it a MILP) |

This is solved as a **mixed-integer linear program** using SciPy's `milp`
(HiGHS solver).

### Infeasibility recovery

If the hard limits can't all be satisfied (e.g. an alert demands a 75% battery
reserve **and** a 300 kW import cap on a 700 kW-load day), the strict model is
infeasible. The runner then automatically re-solves an **elastic** version where
the SoC floor and import cap may be violated at a heavy penalty
(¥10,000/kWh). The slack values tell you **exactly which limit broke, at what
time, and by how much** — instead of just "infeasible".

---

## Project structure

```
bess-fde-workbench/
├── main.py                          # CLI entry point (start here)
├── requirements.txt
├── core_model/
│   ├── formulation.py               # BessParams defaults, JijModeling formulation, input builder
│   └── solver_runner.py             # SciPy HiGHS MILP solver, strict + elastic fallback, DispatchResult
├── ai_harness/
│   ├── exception_translator.py      # Alert text  ->  SoC-floor / import-cap overrides
│   ├── diagnostic_agent.py          # Results     ->  short operator summary
│   └── .claude/skills/bess-model-verifier/SKILL.md   # Claude Code skill: model verification checklist
└── data/
    ├── prepare_japan_grid_data.py   # Builds the hourly site dataset from raw data
    ├── april_2026_hourly.csv        # Ready-to-use dataset (720 h, April 2026)
    └── raw/
        ├── eria_jukyu_202604_03.csv # TEPCO area supply/demand (MW, 5-min/30-min)
        └── spot_summary_2026.csv    # JEPX spot prices (30-min slots, cp932)
```

---

## Setup

Requires Python 3.10+.

```bash
cd bess-fde-workbench
python3 -m venv .venv          # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt
```

`jijmodeling` is optional. If it's installed the formulation is built with it,
but solving always uses SciPy HiGHS.

---

## How to run

All commands are run from the project root with the venv active.

### 1. Basic run (first 24 hours of the dataset)

```bash
python main.py
```

```
Horizon: 2026-04-01 00:00:00 -> 2026-04-01 23:00:00 (24 h) | Battery 2000 kWh / 500 kW

Status: OPTIMAL (scipy.optimize.milp (HiGHS), 0.104s) | Objective: ¥1,733,925 | Peak Import: 769.4 kW
Energy: Import 18,370 kWh, Export 0 kWh, Charge 1,915 kWh, Discharge 752 kWh
```

### 2. Choose a window

```bash
python main.py --start 2026-04-07 --hours 168      # one week
python main.py --start "2026-04-15 06:00" --hours 12
python main.py --hours 720                          # whole month (slower: binaries scale with T)
```

### 3. Try a different battery / site

```bash
python main.py --capacity-kwh 3000 --max-charge-kw 750 --max-discharge-kw 750
python main.py --grid-import-cap-kw 900 --peak-charge-yen-kw 2000
```

### 4. Apply an operator alert

```bash
python main.py --alert "Typhoon warning, cap grid import at 300 kW"
```

```
Alert overrides:
  - SoC reserve floor >= 1500.0 kWh for hour indices [0..23]
  - Grid import cap <= 300.0 kW for hour indices [0..23]
  - Applied 75% storm/outage reserve heuristic

Status: STRICT INFEASIBLE (0.002s) — hard constraints conflicted under current inputs.
Elastic Recovery: OPTIMAL (0.164s) | Recovery Objective: ¥2,651,241 vs strict | Peak Import: 1406.8 kW
Detected constraint conflicts (top items):
  - Grid import cap exceeded by 54.6 kW at 2026-04-01 01:00:00
  - Grid import cap exceeded by 841.4 kW at 2026-04-01 03:00:00
  ...
```

Add `--no-recovery` to skip the elastic re-solve.

### 5. Save the hourly schedule

```bash
python main.py --start 2026-04-07 --hours 168 --out output/week_schedule.csv
```

Columns: `timestamp, load_kw, solar_kw, import_price_yen_kwh, grid_import_kw,
grid_export_kw, charge_kw, discharge_kw, soc_end_kwh, slack_import_kw, slack_soc_kwh`.
(Slack columns are non-zero only when the elastic recovery was used.)

### All CLI options

| Option | Default | Description |
|---|---|---|
| `--data` | `data/april_2026_hourly.csv` | Hourly input CSV |
| `--start` | first row | Start timestamp |
| `--hours` | `24` | Horizon length |
| `--alert` | – | Operator alert text |
| `--no-recovery` | off | Disable elastic fallback |
| `--out` | – | Write schedule CSV |
| `--capacity-kwh` | 2000 | Battery energy capacity |
| `--max-charge-kw` / `--max-discharge-kw` | 500 / 500 | Battery power limits |
| `--init-soc-kwh` / `--end-soc-min-kwh` | 1000 / 800 | Start SoC / minimum end SoC |
| `--grid-import-cap-kw` / `--grid-export-cap-kw` | 1200 / 400 | Grid connection limits |
| `--peak-charge-yen-kw` | 1400 | Demand charge per kW of peak import |
| `--degradation-yen-kwh` | 2.0 | Wear cost per kWh charged or discharged |

Efficiencies (94% / 94%), slack penalty and `dt` are set in `BessParams`
(`core_model/formulation.py`).

---

## Writing alerts

`translate_exception_text` uses keyword/regex rules (no LLM call by default):

| What you write | Effect |
|---|---|
| `reserve 80%` / `soc 60%` | SoC floor = that % of capacity |
| `reserve 1200 kWh` | SoC floor in kWh |
| `typhoon`, `storm`, `outage`, `台風`, `警報`, `停電` (with no % / kWh given) | SoC floor = 75% of capacity |
| `cap grid import at 300 kW`, `transformer limit 500kw`, `受電 600kW` | Import cap for the window |
| `from 14 to 20`, `at 13:00 to 17:00`, `hours14-20` | Hour window (index within the horizon). Default: 0–23 |

Rule tips:
- Put the number **after** the keyword: `reserve 90%` works, `90% reserve` does not.
- The hour window is an **index into the horizon** (0 = first hour of `--start`),
  not a clock time, and defaults to 0–23 even for longer horizons.
- `"hours 14-20"` (with a space) and a bare `"13:00 to 17:00"` are not recognised;
  use `from 14 to 20`.

---

## Using it from Python

```python
import pandas as pd
from core_model.formulation import BessParams
from core_model.solver_runner import solve_dispatch
from ai_harness.exception_translator import translate_exception_text
from ai_harness.diagnostic_agent import build_operator_brief

df = pd.read_csv("data/april_2026_hourly.csv", parse_dates=["timestamp"])
window = df.iloc[:24]
params = BessParams(capacity_kwh=2500)

strict, recovery = solve_dispatch(window, params)
print(build_operator_brief(strict, recovery, window["timestamp"].astype(str).tolist()))

# Arrays per hour
strict.p_imp_kw, strict.p_ch_kw, strict.p_dis_kw, strict.soc_kwh  # soc has T+1 entries
```

With an alert:

```python
patch = translate_exception_text("reserve 80% from 14 to 20", capacity_kwh=params.capacity_kwh)
soc_min, grid_cap = patch.apply(len(window), 0.10 * params.capacity_kwh, params.grid_import_cap_kw)
strict, recovery = solve_dispatch(window, params, soc_min_kwh=soc_min[:len(window)], grid_import_cap_kw=grid_cap)
```

---

## Rebuilding the dataset

`data/april_2026_hourly.csv` is generated from the two raw files:

```bash
python data/prepare_japan_grid_data.py
python data/prepare_japan_grid_data.py --start-date 2026-04-01 --days 7 --out-csv data/week1.csv
python data/prepare_japan_grid_data.py --wheeling-fee 9.0
```

How the site data is derived:

- **Load**: TEPCO area demand, averaged to hourly and scaled so the month's peak = 1,000 kW.
- **Solar**: TEPCO area solar output, scaled so the month's peak = 600 kW.
- **Import price**: JEPX Tokyo area price (30-min slots averaged to hourly) + wheeling fee (default ¥8.5/kWh).
- **Export price**: `min(0.7 × Tokyo price, ¥10/kWh)`.

To use another month, download the TEPCO `eria_jukyu_*.csv` and JEPX
`spot_summary_*.csv` and point `--area-csv` / `--jepx-csv` at them. To use your own
site data, provide a CSV with the columns `timestamp, load_kw, solar_kw,
import_price_yen_kwh, export_price_yen_kwh` and pass it with `main.py --data`.

---

## Model verification (Claude Code skill)

`ai_harness/.claude/skills/bess-model-verifier/SKILL.md` defines a checklist used
when changing the model: power balance residual < 1e-5, SoC trajectory
consistency, no simultaneous charge/discharge, slack diagnostics for infeasible
cases, and solve-time benchmarks at T = 24 / 168 / 720.

---

## Known limitations

- **Solve time grows with horizon length** because there is one binary per hour.
  Roughly 0.1s for 24h and 2s for 168h. A full month (720h) can take much longer.
- **Perfect foresight.** Load, solar and prices are taken as known for the whole
  horizon. There is no forecasting or rolling re-optimization.
- **The alert translator only matches simple patterns** (see *Writing alerts*).
  The `llm_callable` hook in `translate_exception_text` is a placeholder and is
  not wired up yet.
- **The recovery summary label says "vs strict"** but prints the absolute
  recovery cost, because a strict solve that failed has no cost to compare against.
- **The demand charge is applied once** to the horizon's peak import. For
  horizons that are not a full billing month, compare costs with care.
- **JijModeling is not used for solving.** The problem is built with it when the
  package is installed, but it is always solved with SciPy.
