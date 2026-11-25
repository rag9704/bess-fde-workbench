"""HTTP API for the dispatch optimizer.

Run locally:
    uvicorn service.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive OpenAPI page.
"""

from dataclasses import fields, replace
from functools import lru_cache
import logging
import time
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
import pandas as pd
from pydantic import BaseModel, Field

from core_model.data_validation import DataValidationError
from core_model.formulation import BessParams
from core_model.logging_utils import setup_logging
from core_model.solver_runner import SOLVERS
from service.pipeline import DEFAULT_DATA, load_data, run_scenario, select_window

setup_logging()
logger = logging.getLogger("service.api")

app = FastAPI(
    title="BESS Dispatch Optimizer",
    version="0.2.0",
    description="Cost-optimal hourly battery dispatch for a solar + BESS site (JijModeling + OMMX).",
)

_DEFAULTS = BessParams()


class BatteryParams(BaseModel):
    """Site / battery parameters. Omitted fields use the defaults shown."""

    capacity_kwh: float = Field(_DEFAULTS.capacity_kwh, gt=0)
    max_charge_kw: float = Field(_DEFAULTS.max_charge_kw, ge=0)
    max_discharge_kw: float = Field(_DEFAULTS.max_discharge_kw, ge=0)
    charge_eff: float = Field(_DEFAULTS.charge_eff, gt=0, le=1)
    discharge_eff: float = Field(_DEFAULTS.discharge_eff, gt=0, le=1)
    init_soc_kwh: float = Field(_DEFAULTS.init_soc_kwh, ge=0)
    end_soc_min_kwh: float = Field(_DEFAULTS.end_soc_min_kwh, ge=0)
    grid_import_cap_kw: float = Field(_DEFAULTS.grid_import_cap_kw, ge=0)
    grid_export_cap_kw: float = Field(_DEFAULTS.grid_export_cap_kw, ge=0)
    peak_demand_charge_yen_kw: float = Field(_DEFAULTS.peak_demand_charge_yen_kw, ge=0)
    degradation_cost_yen_kwh: float = Field(_DEFAULTS.degradation_cost_yen_kwh, ge=0)

    def to_params(self) -> BessParams:
        p = replace(BessParams(), **self.model_dump())
        if p.init_soc_kwh > p.capacity_kwh or p.end_soc_min_kwh > p.capacity_kwh:
            raise HTTPException(422, "init_soc_kwh and end_soc_min_kwh must not exceed capacity_kwh")
        return p


class SeriesRow(BaseModel):
    timestamp: str
    load_kw: float
    solar_kw: float
    import_price_yen_kwh: float
    export_price_yen_kwh: float


class OptimizeRequest(BaseModel):
    start: str | None = Field(None, description="Start timestamp within the data; default = first row")
    hours: int = Field(24, ge=1, le=744, description="Horizon length in hours")
    series: list[SeriesRow] | None = Field(
        None, description="Your own hourly site data. If omitted, the bundled April 2026 Tokyo dataset is used."
    )
    battery: BatteryParams = Field(default_factory=BatteryParams)
    alert: str | None = Field(None, description="Operator alert, e.g. 'Typhoon warning, cap grid import at 300 kW'")
    solver: Literal["highs", "scip", "scipy"] = "highs"
    auto_recover: bool = True
    compare: bool = True
    include_schedule: bool = True


class OptimizeResponse(BaseModel):
    status: Literal["OPTIMAL", "RECOVERED", "INFEASIBLE"]
    backend: str
    horizon: dict
    objective_yen: float | None
    peak_import_kw: float | None
    solve_time_sec: float
    brief: str
    overrides: list[str] = []
    violations: list[str] = []
    warnings: list[str] = []
    comparison: list[dict] | None = None
    schedule: list[dict] | None = None


@lru_cache(maxsize=1)
def _default_dataset() -> pd.DataFrame:
    return load_data(DEFAULT_DATA)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    )
    return response


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Opening the base URL in a browser lands on the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "solvers": list(SOLVERS), "params": [f.name for f in fields(BessParams)]}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    # Sync endpoint on purpose: FastAPI runs it in a worker thread, so the CPU-bound
    # solve does not block the event loop.
    if req.series is not None:
        df = pd.DataFrame([r.model_dump() for r in req.series])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df = _default_dataset()
    window = select_window(df, req.start, req.hours)
    if window.empty:
        raise HTTPException(404, f"No data for start={req.start!r}")

    try:
        result = run_scenario(
            window,
            req.battery.to_params(),
            alert=req.alert,
            solver=req.solver,
            auto_recover=req.auto_recover,
            compare=req.compare,
        )
    except DataValidationError as exc:
        raise HTTPException(422, {"message": "Input data failed validation", "issues": exc.issues}) from exc

    final = result.final
    if result.strict.status == "OPTIMAL":
        status = "OPTIMAL"
    elif final is not None:
        status = "RECOVERED"
    else:
        status = "INFEASIBLE"

    timestamps = window["timestamp"].astype(str).tolist()
    comparison = None
    if result.comparison is not None:
        comparison = [
            {"policy": name, **{k: round(float(v), 2) for k, v in row.items()}}
            for name, row in result.comparison.iterrows()
        ]
    schedule = None
    if req.include_schedule and (sched := result.schedule_frame()) is not None:
        schedule = sched.to_dict(orient="records")

    return OptimizeResponse(
        status=status,
        backend=result.strict.backend,
        horizon={"start": timestamps[0], "end": timestamps[-1], "hours": len(window)},
        objective_yen=final.objective_yen if final else None,
        peak_import_kw=final.peak_import_kw if final else None,
        solve_time_sec=round(
            result.strict.solve_time_sec + (result.recovery.solve_time_sec if result.recovery else 0), 4
        ),
        brief=result.brief,
        overrides=result.patch.diff_summary() if result.patch else [],
        violations=result.recovery.infeasibility_summary(timestamps) if result.recovery else [],
        warnings=result.warnings,
        comparison=comparison,
        schedule=schedule,
    )
