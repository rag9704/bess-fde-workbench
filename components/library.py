"""Reusable building blocks. The same six blocks model a battery site and a factory warehouse.

| Block       | Energy site (BESS + PV)              | Production / inventory planning              |
|-------------|--------------------------------------|----------------------------------------------|
| Node        | site electrical bus                  | finished-goods stock point                   |
| Profile     | load (out), PV generation (in)       | customer demand (out), with lost-sales cost  |
| Market      | grid import / export at JEPX prices  | contract supplier (buy only)                 |
| Storage     | battery: SoC, efficiencies, wear     | warehouse: inventory, holding cost           |
| Producer    | (e.g. a generator with start cost)   | production line with setup cost              |
| PeakCharge  | demand charge on peak grid import    | capacity fee on peak supplier volume         |

The battery state-of-charge equation and the inventory balance are the same equation:
    level[t+1] = level[t] + eff_in * inflow[t] * dt - outflow[t] * dt / eff_out
and the battery's charge/discharge mode lock and a production setup are the same binary big-M pattern.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .core import ConfigError, ModelBuilder, SeriesSpec, series_values

BIG = 1e6  # bound for otherwise unbounded quantities (slacks, purchases); never binding in practice


@dataclass
class Component:
    name: str
    kind: ClassVar[str] = ""

    def build(self, mb: ModelBuilder) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def data(self, df: pd.DataFrame, state: dict[str, float]) -> dict[str, Any]:
        return {}

    def cost(self, v: dict[str, np.ndarray], d: dict[str, Any], dt: float) -> float:
        return 0.0

    def p(self, key: str) -> str:
        return f"{self.name}_{key}"


@dataclass
class Node(Component):
    """A balance point: everything that flows in must flow out, at every step."""

    kind: ClassVar[str] = "node"

    def build(self, mb: ModelBuilder) -> None:
        mb.add_node(self.name)


@dataclass
class Profile(Component):
    """A fixed time series into or out of a node (load, PV, customer demand).

    With `shortfall_penalty`, an outgoing profile may be left partly unserved at that cost per unit
    (lost sales, load shedding); otherwise it must be served in full.
    """

    node: str = ""
    column: SeriesSpec = 0.0
    direction: str = "out"
    scale: float = 1.0
    shortfall_penalty: float | None = None
    kind: ClassVar[str] = "profile"

    def build(self, mb: ModelBuilder) -> None:
        if self.direction not in ("in", "out"):
            raise ConfigError(f"{self.name}: direction must be 'in' or 'out'")
        value = mb.series(self.name, "value", "Fixed profile [units/h]")
        mb.register_flow(f"{self.name}.value", value)
        if self.shortfall_penalty is None or self.direction == "in":
            mb.connect(self.node, lambda t: value[t], self.direction)
            return
        unmet = mb.problem.ContinuousVar(self.p("unmet"), shape=(mb.T,), lower_bound=0, upper_bound=BIG)
        mb.register_flow(f"{self.name}.unmet", unmet)
        mb.add_constraint(self.p("unmet_le_value"), lambda t: unmet[t] <= value[t], domain=mb.T)
        mb.connect(self.node, lambda t: value[t] - unmet[t], "out")
        pen = mb.scalar(self.name, "penalty", "Cost per unserved unit")
        mb.add_cost(jm_sum(mb, lambda t: pen * unmet[t] * mb.dt))

    def data(self, df, state):
        d = {self.p("value"): self.scale * series_values(df, self.column, self.name)}
        if self.shortfall_penalty is not None and self.direction == "out":
            d[self.p("penalty")] = float(self.shortfall_penalty)
        return d

    def cost(self, v, d, dt):
        if self.p("unmet") not in v:
            return 0.0
        return float(d[self.p("penalty")] * v[self.p("unmet")].sum() * dt)


@dataclass
class Market(Component):
    """Buy from (and optionally sell to) an outside party at time-varying prices (grid, supplier).

    `buy_limit` is an operational cap per step (contract / transformer limit), `sell_max` a hard export cap.
    """

    node: str = ""
    buy_price: SeriesSpec = 0.0
    sell_price: SeriesSpec | None = None
    buy_limit: SeriesSpec | None = None
    sell_max: float = 0.0
    kind: ClassVar[str] = "market"

    def build(self, mb: ModelBuilder) -> None:
        buy = mb.problem.ContinuousVar(self.p("buy"), shape=(mb.T,), lower_bound=0, upper_bound=BIG)
        price_b = mb.series(self.name, "buy_price", "Purchase price [cost/unit]")
        mb.register_flow(f"{self.name}.buy", buy)
        mb.connect(self.node, lambda t: buy[t], "in")
        mb.add_cost(jm_sum(mb, lambda t: price_b[t] * buy[t] * mb.dt))
        if self.buy_limit is not None:
            limit = mb.series(self.name, "buy_limit", "Purchase limit per step [units/h]")
            mb.add_constraint(self.p("buy_limit"), lambda t: buy[t] <= limit[t], domain=mb.T)
        if self.sell_price is not None and self.sell_max > 0:
            sell_max = mb.scalar(self.name, "sell_max", "Sale limit [units/h]")
            sell = mb.problem.ContinuousVar(self.p("sell"), shape=(mb.T,), lower_bound=0, upper_bound=sell_max)
            price_s = mb.series(self.name, "sell_price", "Sale price [revenue/unit]")
            mb.register_flow(f"{self.name}.sell", sell)
            mb.connect(self.node, lambda t: sell[t], "out")
            mb.add_cost(jm_sum(mb, lambda t: -price_s[t] * sell[t] * mb.dt))

    def _sells(self) -> bool:
        return self.sell_price is not None and self.sell_max > 0

    def data(self, df, state):
        d = {self.p("buy_price"): series_values(df, self.buy_price, self.name)}
        if self.buy_limit is not None:
            d[self.p("buy_limit")] = series_values(df, self.buy_limit, self.name)
        if self._sells():
            d[self.p("sell_price")] = series_values(df, self.sell_price, self.name)
            d[self.p("sell_max")] = float(self.sell_max)
        return d

    def cost(self, v, d, dt):
        c = float((d[self.p("buy_price")] * v[self.p("buy")]).sum() * dt)
        if self._sells():
            c -= float((d[self.p("sell_price")] * v[self.p("sell")]).sum() * dt)
        return c


@dataclass
class Storage(Component):
    """A stock that carries quantity between steps: battery, warehouse, tank.

    level[t+1] = level[t] + eff_in * in[t] * dt - out[t] * dt / eff_out
    `exclusive_modes` forbids filling and emptying in the same step (one binary per step), as a battery
    inverter cannot charge and discharge at once. `floor` may be a constant or a column (reserve plan).
    """

    node: str = ""
    capacity: float = 0.0
    max_in: float = BIG
    max_out: float = BIG
    eff_in: float = 1.0
    eff_out: float = 1.0
    initial: float = 0.0
    final_min: float = 0.0
    floor: SeriesSpec = 0.0
    throughput_cost: float = 0.0
    holding_cost: float = 0.0
    exclusive_modes: bool = False
    kind: ClassVar[str] = "storage"

    def build(self, mb: ModelBuilder) -> None:
        P = mb.problem
        cap = mb.scalar(self.name, "capacity", "Capacity [units]")
        max_in = mb.scalar(self.name, "max_in", "Max fill rate [units/h]")
        max_out = mb.scalar(self.name, "max_out", "Max draw rate [units/h]")
        eff_in = mb.scalar(self.name, "eff_in", "Fill efficiency")
        eff_out = mb.scalar(self.name, "eff_out", "Draw efficiency")
        init = mb.scalar(self.name, "initial", "Initial level [units]")
        final_min = mb.scalar(self.name, "final_min", "Minimum final level [units]")
        floor = mb.states(self.name, "floor", "Minimum level per state [units]")

        level = P.ContinuousVar(self.p("level"), shape=(mb.T + 1,), lower_bound=0, upper_bound=cap)
        fill = P.ContinuousVar(self.p("in"), shape=(mb.T,), lower_bound=0, upper_bound=max_in)
        draw = P.ContinuousVar(self.p("out"), shape=(mb.T,), lower_bound=0, upper_bound=max_out)
        mb.register_flow(f"{self.name}.in", fill)
        mb.register_flow(f"{self.name}.out", draw)
        mb.register_flow(f"{self.name}.level", level)
        mb.connect(self.node, lambda t: fill[t], "out")  # filling takes from the node
        mb.connect(self.node, lambda t: draw[t], "in")  # drawing supplies the node

        dt = mb.dt
        mb.add_constraint(
            self.p("dynamics"),
            lambda t: level[t + 1] == level[t] + eff_in * fill[t] * dt - (1.0 / eff_out) * draw[t] * dt,
            domain=mb.T,
        )
        mb.add_constraint(self.p("initial"), level[0] == init)
        mb.add_constraint(self.p("final_min"), level[mb.T] >= final_min)
        mb.add_constraint(self.p("floor"), lambda k: level[k] >= floor[k], domain=mb.T + 1)
        if self.exclusive_modes:
            z = P.BinaryVar(self.p("filling"), shape=(mb.T,), description="1 = filling mode, 0 = drawing mode")
            mb.add_constraint(self.p("fill_mode"), lambda t: fill[t] <= max_in * z[t], domain=mb.T)
            mb.add_constraint(self.p("draw_mode"), lambda t: draw[t] <= max_out * (1 - z[t]), domain=mb.T)
        if self.throughput_cost:
            c = mb.scalar(self.name, "throughput_cost", "Cost per unit moved in or out")
            mb.add_cost(jm_sum(mb, lambda t: c * (fill[t] + draw[t]) * dt))
        if self.holding_cost:
            h = mb.scalar(self.name, "holding_cost", "Cost per unit held per hour/period")
            mb.add_cost(jm_sum(mb, lambda t: h * level[t + 1] * dt))

    def data(self, df, state):
        floor = series_values(df, self.floor, self.name)
        initial = float(state.get(f"{self.name}.level", self.initial))
        d = {
            self.p("capacity"): float(self.capacity),
            self.p("max_in"): float(self.max_in),
            self.p("max_out"): float(self.max_out),
            self.p("eff_in"): float(self.eff_in),
            self.p("eff_out"): float(self.eff_out),
            self.p("initial"): initial,
            self.p("final_min"): float(self.final_min),
            self.p("floor"): np.append(floor, floor[-1] if len(floor) else 0.0),
        }
        if self.throughput_cost:
            d[self.p("throughput_cost")] = float(self.throughput_cost)
        if self.holding_cost:
            d[self.p("holding_cost")] = float(self.holding_cost)
        return d

    def cost(self, v, d, dt):
        c = self.throughput_cost * float((v[self.p("in")] + v[self.p("out")]).sum()) * dt
        return c + self.holding_cost * float(v[self.p("level")][1:].sum()) * dt


@dataclass
class Producer(Component):
    """Controllable supply with a per-unit cost and, optionally, a setup (start) cost and minimum run rate.

    With `setup_cost` or `min_output`, a binary per step decides whether the producer runs:
        output[t] <= capacity * on[t],  output[t] >= min_output * on[t]
    """

    node: str = ""
    capacity: float = 0.0
    unit_cost: SeriesSpec = 0.0
    setup_cost: float = 0.0
    min_output: float = 0.0
    kind: ClassVar[str] = "producer"

    def build(self, mb: ModelBuilder) -> None:
        P = mb.problem
        cap = mb.scalar(self.name, "capacity", "Max output [units/h]")
        out = P.ContinuousVar(self.p("output"), shape=(mb.T,), lower_bound=0, upper_bound=cap)
        cost = mb.series(self.name, "unit_cost", "Cost per unit produced")
        mb.register_flow(f"{self.name}.output", out)
        mb.connect(self.node, lambda t: out[t], "in")
        mb.add_cost(jm_sum(mb, lambda t: cost[t] * out[t] * mb.dt))
        if self.setup_cost or self.min_output:
            on = P.BinaryVar(self.p("on"), shape=(mb.T,), description="1 = producer runs in this step")
            mb.add_constraint(self.p("run_if_on"), lambda t: out[t] <= cap * on[t], domain=mb.T)
            if self.min_output:
                mn = mb.scalar(self.name, "min_output", "Minimum output when running")
                mb.add_constraint(self.p("min_run"), lambda t: out[t] >= mn * on[t], domain=mb.T)
            if self.setup_cost:
                sc = mb.scalar(self.name, "setup_cost", "Cost per step in which the producer runs")
                mb.add_cost(jm_sum(mb, lambda t: sc * on[t]))

    def _binary(self) -> bool:
        return bool(self.setup_cost or self.min_output)

    def data(self, df, state):
        d = {
            self.p("capacity"): float(self.capacity),
            self.p("unit_cost"): series_values(df, self.unit_cost, self.name),
        }
        if self.min_output:
            d[self.p("min_output")] = float(self.min_output)
        if self.setup_cost:
            d[self.p("setup_cost")] = float(self.setup_cost)
        return d

    def cost(self, v, d, dt):
        c = float((d[self.p("unit_cost")] * v[self.p("output")]).sum() * dt)
        if self.setup_cost:
            c += self.setup_cost * float(np.round(v[self.p("on")]).sum())
        return c


@dataclass
class PeakCharge(Component):
    """A charge per unit of the highest value a flow reaches (demand charge, reserved capacity fee).

    `prior_peak` (or the rolling state "<name>.peak") is the peak already reached in the billing period:
    staying below it is free, raising it costs `rate` per unit.
    """

    flow: str = ""
    rate: float = 0.0
    prior_peak: float = 0.0
    kind: ClassVar[str] = "peak_charge"

    def build(self, mb: ModelBuilder) -> None:
        target = mb.flow(self.flow)
        prior = mb.scalar(self.name, "prior_peak", "Peak already reached in the billing period")
        rate = mb.scalar(self.name, "rate", "Charge per unit of peak")
        peak = mb.problem.ContinuousVar(self.p("peak"), lower_bound=prior, upper_bound=BIG)
        mb.add_constraint(self.p("def"), lambda t: peak >= target[t], domain=mb.T)
        mb.add_cost(rate * peak)

    def data(self, df, state):
        return {
            self.p("rate"): float(self.rate),
            self.p("prior_peak"): float(state.get(f"{self.name}.peak", self.prior_peak)),
        }

    def cost(self, v, d, dt):
        return self.rate * float(v[self.p("peak")])


def jm_sum(mb: ModelBuilder, fn) -> Any:
    import jijmodeling as jm

    return jm.sum(mb.T, fn)


COMPONENT_TYPES: dict[str, type[Component]] = {
    c.kind: c for c in (Node, Profile, Market, Storage, Producer, PeakCharge)
}
