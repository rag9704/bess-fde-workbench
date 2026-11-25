"""One end-to-end scenario run, shared by the CLI (main.py) and the HTTP API.

validate data -> (optional) translate operator alert -> solve strict / elastic
-> score against baselines -> build operator brief + hourly schedule.
"""

from dataclasses import dataclass, field, replace
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ai_harness.diagnostic_agent import build_operator_brief
from ai_harness.exception_translator import ScheduleOverridePatch
from ai_harness.llm_explainer import explain_conflicts
from ai_harness.llm_translator import TranslatorMode, translate_alert
from core_model.data_validation import validate_timeseries
from core_model.evaluation import Schedule, compare_policies, no_battery_schedule, rule_based_schedule
from core_model.formulation import BessParams
from core_model.solver_runner import DispatchResult, SolverOptions, solve_dispatch

logger = logging.getLogger(__name__)

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "april_2026_hourly.csv"


@dataclass
class ScenarioResult:
    window: pd.DataFrame
    params: BessParams
    strict: DispatchResult
    recovery: DispatchResult | None
    patch: ScheduleOverridePatch | None
    comparison: pd.DataFrame | None
    brief: str
    warnings: list[str] = field(default_factory=list)
    translator_source: str | None = None  # "rules" or "llm" when an alert was given
    explanation: str | None = None  # optional Claude-written operator explanation

    @property
    def final(self) -> DispatchResult | None:
        """The schedule to act on: strict if feasible, else the elastic recovery (if any)."""
        if self.strict.usable:
            return self.strict
        if self.recovery is not None and self.recovery.usable:
            return self.recovery
        return None

    def schedule_frame(self) -> pd.DataFrame | None:
        res = self.final
        if res is None:
            return None
        w, T = self.window, len(self.window)
        return pd.DataFrame(
            {
                "timestamp": w["timestamp"].astype(str),
                "load_kw": w["load_kw"],
                "solar_kw": w["solar_kw"],
                "import_price_yen_kwh": w["import_price_yen_kwh"],
                "grid_import_kw": np.round(res.p_imp_kw, 1),
                "grid_export_kw": np.round(res.p_exp_kw, 1),
                "charge_kw": np.round(res.p_ch_kw, 1),
                "discharge_kw": np.round(res.p_dis_kw, 1),
                "soc_end_kwh": np.round(res.soc_kwh[1 : T + 1], 1),
                "slack_import_kw": np.round(res.slack_imp_kw, 1),
                "slack_soc_kwh": np.round(res.slack_soc_kwh[1 : T + 1], 1),
            }
        )


def load_data(path: Path | str = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def select_window(df: pd.DataFrame, start: str | None, hours: int) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    return df.iloc[:hours].reset_index(drop=True)


def run_scenario(
    window: pd.DataFrame,
    params: BessParams,
    alert: str | None = None,
    solver: str = "highs",
    auto_recover: bool = True,
    compare: bool = True,
    translator: TranslatorMode = "rules",
    solver_options: SolverOptions | None = None,
    explain: bool = False,
    llm_client=None,
) -> ScenarioResult:
    warnings = validate_timeseries(window, expected_step_hours=params.dt_hours)
    T = len(window)
    timestamps = window["timestamp"].astype(str).tolist()

    patch = None
    source = None
    soc_min = grid_cap = None
    if alert:
        patch, source = translate_alert(
            alert,
            capacity_kwh=params.capacity_kwh,
            horizon_hours=T,
            horizon_start=timestamps[0],
            mode=translator,
            client=llm_client,
        )
        if patch.end_soc_min_kwh is not None:
            # Overrides may only tighten limits, never loosen them.
            params = replace(params, end_soc_min_kwh=max(params.end_soc_min_kwh, patch.end_soc_min_kwh))
        soc_min_full, grid_cap = patch.apply(
            T, params.default_soc_min_frac * params.capacity_kwh, params.grid_import_cap_kw
        )
        soc_min = soc_min_full
        logger.info("alert translated", extra={"alert": alert, "source": source, "overrides": patch.diff_summary()})

    strict, recovery = solve_dispatch(
        window,
        params,
        soc_min_kwh=soc_min,
        grid_import_cap_kw=grid_cap,
        auto_recover_infeasible=auto_recover,
        solver=solver,
        options=solver_options,
    )

    result = ScenarioResult(
        window=window,
        params=params,
        strict=strict,
        recovery=recovery,
        patch=patch,
        comparison=None,
        brief=build_operator_brief(strict, recovery, timestamps),
        warnings=warnings,
        translator_source=source,
    )
    if explain and recovery is not None:
        result.explanation = explain_conflicts(
            alert=alert,
            overrides=patch.diff_summary() if patch else [],
            violations=recovery.infeasibility_summary(timestamps),
            params=params,
            client=llm_client,
        )

    final = result.final
    if compare and patch is not None:
        # Baseline policies don't model alert overrides (reserve floors / temporary caps),
        # so a side-by-side cost comparison would not be like-for-like.
        result.warnings.append("Baseline comparison skipped: baselines do not model alert overrides")
    elif compare and final is not None:
        label = "Optimized (MILP)" if final is strict else "Optimized (elastic recovery)"
        opt = Schedule(label, final.p_imp_kw, final.p_exp_kw, final.p_ch_kw, final.p_dis_kw, final.soc_kwh)
        result.comparison = compare_policies(
            window,
            params,
            [no_battery_schedule(window, params), rule_based_schedule(window, params), opt],
            grid_import_cap_kw=grid_cap,
        )
    return result
