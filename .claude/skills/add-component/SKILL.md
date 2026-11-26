---
name: add-component
description: Add a new reusable building block (asset type, tariff, constraint pattern) to components/library.py, or model a new customer site or a non-energy problem in YAML under sites/. Covers the component contract, naming rules, the tests a new block needs, and how to prove the new model is right.
---

# Add a component or a site

The library in `components/` turns a YAML site description into one JijModeling model. A new customer
should normally need **only a YAML file**. Add Python only when no existing block fits.

## First: can existing blocks express it?

| Need | Use |
|---|---|
| Fixed series in or out (load, PV, orders, forecasts) | `profile` (`shortfall_penalty` makes unmet demand allowed at a cost) |
| Buy/sell at a price, with a per-step limit | `market` |
| Anything that carries quantity between steps | `storage` (`exclusive_modes` for batteries, `holding_cost` for stock) |
| Controllable supply with start/setup cost or minimum run | `producer` |
| Charge on the peak of a flow (demand charge, capacity fee) | `peak_charge` |
| Several balance points (e.g. DC bus + AC bus, two warehouses) | several `node`s; connect them with a `storage` or a lossless `producer` |

Write the YAML, then run `python -m components sites/<site>.yaml --steps 24 --show <component>.<var>`.

## Adding a block (`components/library.py`)

A component is a dataclass subclassing `Component` with a unique `kind` and three methods:

1. `build(mb)`: declare placeholders with `mb.series` / `mb.states` / `mb.scalar` (names are prefixed with the
   component name automatically), variables with `mb.problem.ContinuousVar(self.p("x"), ...)`, connect flows
   with `mb.connect(node, lambda t: x[t], "in" | "out")`, register flows other blocks may reference with
   `mb.register_flow(f"{self.name}.x", x)`, add costs with `mb.add_cost(...)`.
2. `data(df, state)`: return the placeholder values. Read columns with `series_values(df, spec, self.name)` so a
   field can be a constant or a column. Read rolling state (`state.get(f"{self.name}.level", default)`).
3. `cost(values, data, dt)`: the same cost in numpy, for the per-component breakdown.

JijModeling rules that bite:
- Use `jm.sum(mb.T, lambda t: ...)` and `mb.add_constraint(name, lambda t: ..., domain=mb.T)`. No Python `sum`,
  comprehensions or generators over JijModeling objects.
- **Never give a constraint lambda default arguments** (`lambda t, x=x: ...`): JijModeling binds every
  parameter to an index. Use a closure (see `_balance` in `components/core.py`).
- Bounds must be finite. Use `BIG` for "unbounded".

Register the class in `COMPONENT_TYPES`; config validation (unknown fields, bad names) then works automatically.

## Prove it is right

Add to `tests/test_components.py`:
- **Physics:** the block's own equations hold in a solved example (balance, dynamics, bounds, binaries).
- **An independent check** whenever one exists: a closed-form answer, a small DP or brute force (see
  `wagner_whitin`), or equality with a hand-written model (see `test_bess_site_matches_hand_written_model`).
- **Cost:** `result.objective == sum(result.costs.values())`, so `cost()` and the model agree.
- **Config errors** for any new required field.

Then run `pytest -q` and `python verify.py --quick`. If the block changes the battery model, the
`bess-model-verifier` skill applies too.
