"""A site = components + data. Load it from YAML, build the JijModeling model, solve through OMMX.

site = load_site("sites/tokyo_office_bess.yaml")
result = solve_site(site, site.load_data().iloc[:24])
print(result.summary())
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core_model.solver_runner import Infeasible, NoSolution, SolverOptions

from .core import ConfigError, ModelBuilder
from .library import COMPONENT_TYPES, Component


@dataclass
class Site:
    name: str
    components: list[Component]
    dt_hours: float = 1.0
    description: str = ""
    data_path: Path | None = None
    _problem: Any = field(default=None, init=False, repr=False)

    def problem(self):
        """The assembled JijModeling problem (built once per site, re-used for any horizon)."""
        if self._problem is None:
            mb = ModelBuilder(self.name)
            for c in self.components:
                c.build(mb)
            self._problem = mb.finalize()
        return self._problem

    def load_data(self) -> pd.DataFrame:
        if self.data_path is None:
            raise ConfigError(f"site {self.name!r} has no 'data' file")
        df = pd.read_csv(self.data_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def instance_data(self, df: pd.DataFrame, state: dict[str, float] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"T": len(df), "dt": float(self.dt_hours)}
        for c in self.components:
            data.update(c.data(df.reset_index(drop=True), state or {}))
        return data

    def component(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)


def site_from_dict(cfg: dict[str, Any], base_dir: Path | None = None) -> Site:
    """Build a Site from a parsed YAML/JSON description. Errors name the offending component and field."""
    if not isinstance(cfg, dict) or "components" not in cfg:
        raise ConfigError("a site needs a 'components' list")
    comps: list[Component] = []
    names: set[str] = set()
    for i, raw in enumerate(cfg["components"]):
        raw = dict(raw)
        kind = raw.pop("type", None)
        if kind not in COMPONENT_TYPES:
            raise ConfigError(f"component #{i + 1}: unknown type {kind!r}; use one of {', '.join(COMPONENT_TYPES)}")
        cls = COMPONENT_TYPES[kind]
        allowed = {f.name for f in fields(cls) if f.init}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigError(
                f"component {raw.get('name', i + 1)!r} ({kind}): unknown field(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        name = raw.get("name")
        if not name or not str(name).isidentifier():
            raise ConfigError(f"component #{i + 1} ({kind}): 'name' must be an identifier, got {name!r}")
        if name in names:
            raise ConfigError(f"duplicate component name {name!r}")
        names.add(name)
        comps.append(cls(**raw))
    data = cfg.get("data")
    data_path = (base_dir / data).resolve() if data and base_dir else (Path(data) if data else None)
    return Site(
        name=cfg.get("site", "site"),
        components=comps,
        dt_hours=float(cfg.get("dt_hours", 1.0)),
        description=cfg.get("description", ""),
        data_path=data_path,
    )


def load_site(path: str | Path) -> Site:
    path = Path(path)
    return site_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")), base_dir=path.parent)


# --------------------------------------------------------------------------- solving


@dataclass
class SiteResult:
    site: Site
    status: str
    objective: float | None
    values: dict[str, np.ndarray]
    costs: dict[str, float]
    data: dict[str, Any]
    timings: dict[str, float]

    def get(self, key: str) -> np.ndarray:
        """A decision or input series by "component.name" (e.g. "battery.level", "load.value")."""
        comp, var = key.split(".", 1)
        name = f"{comp}_{var}"
        if name in self.values:
            return self.values[name]
        if name in self.data:
            return np.asarray(self.data[name], dtype=float)
        raise KeyError(f"{key!r} is not a variable or input of this site")

    def summary(self) -> str:
        lines = [f"{self.site.name}: {self.status}, total cost {self.objective:,.0f}"]
        for name, c in self.costs.items():
            if abs(c) > 1e-9:
                lines.append(f"  {name:<20} {c:>14,.0f}")
        lines.append(f"  (compile {self.timings['compile_sec']:.2f}s, solve {self.timings['solve_sec']:.2f}s)")
        return "\n".join(lines)


def _values(instance, col) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[tuple[int, ...], float]]] = {}
    for i, var in enumerate(instance.used_decision_variables):
        grouped.setdefault(var.name, []).append((tuple(var.subscripts), float(col[i])))
    out = {}
    for name, items in grouped.items():
        items.sort()
        out[name] = np.array(items[0][1]) if items[0][0] == () else np.array([v for _, v in items])
    return out


def solve_site(
    site: Site,
    df: pd.DataFrame,
    state: dict[str, float] | None = None,
    solver: str = "highs",
    options: SolverOptions | None = None,
) -> SiteResult:
    """Solve the site over `df`. `state` carries rolling values ("battery.level", "demand_charge.peak")."""
    opts = options or SolverOptions()
    t0 = time.perf_counter()
    data = site.instance_data(df, state)
    instance = site.problem().eval(data)
    t1 = time.perf_counter()
    if solver == "highs":
        values, status = _solve_highs(instance, opts)
    elif solver == "scip":
        values, status = _solve_scip(instance, opts)
    else:
        raise ConfigError(f"solver {solver!r} not supported for component sites (use highs or scip)")
    t2 = time.perf_counter()
    dt = float(site.dt_hours)
    costs = {c.name: c.cost(values, data, dt) for c in site.components}
    return SiteResult(
        site, status, sum(costs.values()), values, costs, data, {"compile_sec": t1 - t0, "solve_sec": t2 - t1}
    )


def _solve_highs(instance, opts: SolverOptions):
    import highspy

    from core_model.solver_runner import _new_highs_adapter

    h = _new_highs_adapter(instance).solver_input
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", float(opts.mip_rel_gap if opts.mip_rel_gap is not None else 1e-9))
    if opts.time_limit_sec is not None:
        h.setOptionValue("time_limit", float(opts.time_limit_sec))
    h.run()
    ms = h.getModelStatus()
    if ms in (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnboundedOrInfeasible):
        raise Infeasible
    if h.getInfo().primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
        raise NoSolution(h.modelStatusToString(ms))
    status = "OPTIMAL" if ms == highspy.HighsModelStatus.kOptimal else "FEASIBLE"
    return _values(instance, np.asarray(h.getSolution().col_value)), status


def _solve_scip(instance, opts: SolverOptions):
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

    adapter = OMMXPySCIPOptAdapter(instance)
    m = adapter.solver_input
    m.hideOutput()
    m.setParam("limits/gap", float(opts.mip_rel_gap if opts.mip_rel_gap is not None else 1e-9))
    if opts.time_limit_sec is not None:
        m.setParam("limits/time", float(opts.time_limit_sec))
    m.optimize()
    if m.getStatus() in ("infeasible", "inforunbd"):
        raise Infeasible
    if m.getNSols() == 0:
        raise NoSolution(m.getStatus())
    solution = adapter.decode(m)
    col = [solution.state.entries[v.id] for v in instance.used_decision_variables]
    return _values(instance, col), "OPTIMAL" if m.getStatus() in ("optimal", "gaplimit") else "FEASIBLE"
