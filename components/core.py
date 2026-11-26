"""Model assembly: components add placeholders, variables, flows, constraints and costs to one JijModeling problem.

A site is a list of components. Each component
  * declares its data (placeholders) and decisions (variables) in `build()`,
  * connects flows to **nodes** (`ModelBuilder.connect`); every node gets a balance constraint
    "sum of flows in == sum of flows out" at every time step,
  * adds its own costs to the objective,
  * turns a time-series DataFrame into its placeholder values in `data()`.

Because placeholders are symbolic, the assembled model is built once per site layout and
re-used for any horizon or data (only `problem.eval(data)` changes).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jijmodeling as jm
import numpy as np
import pandas as pd

FlowFn = Callable[[Any], Any]  # t -> flow expression at step t (units per hour)
SeriesSpec = float | int | str  # a constant, or the name of a data column


class ConfigError(ValueError):
    """A site description is invalid (unknown component, missing column, bad reference, ...)."""


@dataclass
class ModelBuilder:
    name: str
    problem: jm.Problem = field(init=False)
    T: Any = field(init=False)
    dt: Any = field(init=False)
    flows: dict[str, Any] = field(default_factory=dict)  # "component.flow" -> decision variable / placeholder
    _node_in: dict[str, list[FlowFn]] = field(default_factory=dict)
    _node_out: dict[str, list[FlowFn]] = field(default_factory=dict)
    _costs: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.problem = jm.Problem(self.name, sense=jm.ProblemSense.MINIMIZE)
        self.T = self.problem.Natural("T", description="Number of time steps")
        self.dt = self.problem.Float("dt", description="Step length [h or period]")

    # -- declarations ------------------------------------------------------------------------
    def series(self, owner: str, key: str, description: str = "") -> Any:
        return self.problem.Float(f"{owner}_{key}", shape=(self.T,), description=description)

    def states(self, owner: str, key: str, description: str = "") -> Any:
        return self.problem.Float(f"{owner}_{key}", shape=(self.T + 1,), description=description)

    def scalar(self, owner: str, key: str, description: str = "") -> Any:
        return self.problem.Float(f"{owner}_{key}", description=description)

    def add_node(self, node: str) -> None:
        self._node_in.setdefault(node, [])
        self._node_out.setdefault(node, [])

    def connect(self, node: str, flow: FlowFn, direction: str) -> None:
        """direction="in": the flow enters the node (supply); "out": it leaves the node (demand)."""
        if node not in self._node_in:
            raise ConfigError(f"unknown node {node!r}; declare it with a 'node' component first")
        (self._node_in if direction == "in" else self._node_out)[node].append(flow)

    def register_flow(self, key: str, var: Any) -> None:
        self.flows[key] = var

    def flow(self, key: str) -> Any:
        if key not in self.flows:
            raise ConfigError(f"unknown flow {key!r}; available: {', '.join(sorted(self.flows))}")
        return self.flows[key]

    def add_cost(self, expr: Any) -> None:
        self._costs.append(expr)

    def add_constraint(self, name: str, expr: Any, domain: Any = None) -> None:
        self.problem += self.problem.Constraint(name, expr, domain=domain)

    # -- assembly ----------------------------------------------------------------------------
    def finalize(self) -> jm.Problem:
        for node in self._node_in:
            ins, outs = self._node_in[node], self._node_out[node]
            if not ins or not outs:
                raise ConfigError(f"node {node!r} needs at least one flow in and one flow out")

            self.add_constraint(f"balance_{node}", _balance(ins, outs), domain=self.T)
        if not self._costs:
            raise ConfigError("the site has no costs to minimise")
        self.problem += _total(self._costs)
        return self.problem


def series_values(df: pd.DataFrame, spec: SeriesSpec, what: str) -> np.ndarray:
    """A column of `df`, or a constant broadcast to len(df)."""
    if isinstance(spec, str):
        if spec not in df.columns:
            raise ConfigError(f"{what}: column {spec!r} not in data (columns: {', '.join(df.columns)})")
        return df[spec].to_numpy(dtype=float)
    return np.full(len(df), float(spec))


def _total(terms: list[Any]) -> Any:
    """Add expressions with a plain loop (JijModeling forbids generator-based reductions over its objects)."""
    out = terms[0]
    for term in terms[1:]:
        out = out + term
    return out


def _balance(ins: list[FlowFn], outs: list[FlowFn]) -> FlowFn:
    # A closure, not default arguments: JijModeling binds every parameter of the function to an index.
    def balance(t):
        return _total([f(t) for f in ins]) == _total([f(t) for f in outs])

    return balance
