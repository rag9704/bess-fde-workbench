"""Claude-backed translation of operator alerts into schedule overrides.

Design (the "AI harness" contract):
  * Claude only *extracts* what the alert says, into a typed schema (structured outputs).
  * Deterministic code owns every decision that affects the physical schedule:
      - values are clamped to physical limits (0..capacity, 0..100 %, hours inside the horizon),
      - site policy (e.g. the 75 % storm reserve) is applied by code, not chosen by the model,
      - overrides can only *tighten* limits (ScheduleOverridePatch.apply takes max/min), so a bad
        extraction can make a schedule more conservative but never less safe.
  * Anything Claude could not map to an override is surfaced to the operator, not dropped.
  * If the API is unavailable, refuses, or returns something unusable, the caller falls back
    to the deterministic rule parser (see `translate_alert`).
"""

from dataclasses import dataclass
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from .exception_translator import ScheduleOverridePatch, translate_exception_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
STORM_RESERVE_FRAC = 0.75  # site policy, applied by code


# --------------------------------------------------------------------------- extraction schema


class ReserveRequest(BaseModel):
    start_hour: int = Field(description="First hour index (0 = first hour of the horizon), inclusive")
    end_hour: int = Field(description="Last hour index, inclusive")
    unit: Literal["percent", "kwh"]
    value: float = Field(description="Minimum state of charge to hold, in the given unit")


class ImportCapRequest(BaseModel):
    start_hour: int = Field(description="First hour index (0 = first hour of the horizon), inclusive")
    end_hour: int = Field(description="Last hour index, inclusive")
    cap_kw: float = Field(description="Maximum grid import in kW")


class AlertExtraction(BaseModel):
    """What the alert asks for. Only fill fields the alert actually states."""

    reserves: list[ReserveRequest] = Field(description="Explicit minimum-SoC / reserve requests")
    import_caps: list[ImportCapRequest] = Field(description="Explicit grid import limits (kW)")
    end_soc_min_kwh: float | None = Field(description="Explicit minimum SoC at the END of the horizon, in kWh")
    end_soc_min_percent: float | None = Field(description="Explicit minimum SoC at the END of the horizon, in %")
    storm_or_outage_risk: bool = Field(
        description="True if the alert warns of a typhoon, storm, outage, blackout or similar grid risk"
    )
    storm_window: ImportCapRequest | None = Field(
        description="Hour window of the storm/outage risk if stated (cap_kw is ignored; set it to 0)"
    )
    not_applied: list[str] = Field(
        description="Requests in the alert that are NOT a reserve, import cap or end-of-horizon SoC "
        "(e.g. price changes, export limits, maintenance notes). Quote them briefly."
    )


SYSTEM_PROMPT = """\
You convert operator alerts for a commercial solar + battery site into structured override requests.
Alerts may be in English or Japanese.

The optimization horizon is described in the user message. Hour indices are relative to the first hour of
the horizon: index 0 is the first hour. Convert clock times to indices using the horizon start time.
Windows are inclusive on both ends: "14:00 to 17:00" means start_hour=14, end_hour=17 when the horizon
starts at 00:00. If the alert gives no time window, use the whole horizon.

Extract only what the alert states:
- reserves: "hold 80%", "keep at least 1,200 kWh", "SoC ≥ 60%", "予備率70%", "残量1500kWh以上".
- import_caps: grid import / contract / transformer / 受電 / デマンド limits, in kW. Convert MW to kW.
- end_soc_min_*: only for a requirement on the battery level at the END of the horizon.
- storm_or_outage_risk: set true for typhoon/storm/outage/blackout warnings (台風, 停電, 警報, 雷...).
  Do NOT invent a reserve value for it; site policy decides that.
- not_applied: anything else the operator asked for that does not fit the fields above.
Never guess numbers that are not in the alert. If the alert asks for nothing, return empty lists."""


# --------------------------------------------------------------------------- guardrails


def _clamp_window(start: int, end: int, horizon: int) -> tuple[int, int] | None:
    if start > end:
        start, end = end, start
    start, end = max(0, start), min(horizon - 1, end)
    return (start, end) if start <= end else None


def extraction_to_patch(ex: AlertExtraction, capacity_kwh: float, horizon_hours: int) -> ScheduleOverridePatch:
    """Deterministic, bounded conversion from Claude's extraction to model overrides."""
    patch = ScheduleOverridePatch()

    for r in ex.reserves:
        win = _clamp_window(r.start_hour, r.end_hour, horizon_hours)
        if win is None:
            patch.notes.append(f"Ignored reserve request outside the horizon (hours {r.start_hour}-{r.end_hour})")
            continue
        if r.unit == "percent":
            kwh = capacity_kwh * min(max(r.value, 0.0), 100.0) / 100.0
        else:
            kwh = min(max(r.value, 0.0), capacity_kwh)
            if r.value > capacity_kwh:
                patch.notes.append(f"Reserve {r.value:.0f} kWh exceeds capacity; clamped to {capacity_kwh:.0f} kWh")
        patch.soc_min_overrides.append((win[0], win[1], round(kwh, 1)))

    for c in ex.import_caps:
        win = _clamp_window(c.start_hour, c.end_hour, horizon_hours)
        if win is None:
            patch.notes.append(f"Ignored import cap outside the horizon (hours {c.start_hour}-{c.end_hour})")
            continue
        if c.cap_kw < 0:
            patch.notes.append(f"Ignored negative import cap ({c.cap_kw} kW)")
            continue
        patch.grid_cap_overrides.append((win[0], win[1], float(c.cap_kw)))

    end_vals = []
    if ex.end_soc_min_kwh is not None:
        end_vals.append(min(max(ex.end_soc_min_kwh, 0.0), capacity_kwh))
    if ex.end_soc_min_percent is not None:
        end_vals.append(capacity_kwh * min(max(ex.end_soc_min_percent, 0.0), 100.0) / 100.0)
    if end_vals:
        patch.end_soc_min_kwh = round(max(end_vals), 1)

    if ex.storm_or_outage_risk and not ex.reserves:
        win = (0, horizon_hours - 1)
        if ex.storm_window is not None:
            win = _clamp_window(ex.storm_window.start_hour, ex.storm_window.end_hour, horizon_hours) or win
        patch.soc_min_overrides.append((win[0], win[1], round(STORM_RESERVE_FRAC * capacity_kwh, 1)))
        patch.notes.append("Applied 75% storm/outage reserve heuristic")

    for item in ex.not_applied:
        patch.notes.append(f"Not applied (no matching control): {item}")
    return patch


# --------------------------------------------------------------------------- Claude call


class LLMUnavailable(Exception):
    """Claude could not produce a usable extraction (no credentials, API error, refusal, ...)."""


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


def extract_with_claude(
    text: str,
    capacity_kwh: float,
    horizon_hours: int,
    horizon_start: str | None = None,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[AlertExtraction, LLMUsage]:
    """Ask Claude for a structured extraction of the alert. Raises LLMUnavailable on any failure."""
    import anthropic

    try:
        client = client or anthropic.Anthropic()
    except anthropic.AnthropicError as exc:  # e.g. no credentials configured
        raise LLMUnavailable(str(exc)) from exc

    user = (
        f"Horizon: {horizon_hours} hourly steps starting {horizon_start or '00:00 (hour index = clock hour)'}.\n"
        f"Battery capacity: {capacity_kwh:.0f} kWh.\n\n"
        f"Operator alert:\n<alert>\n{text}\n</alert>"
    )
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=AlertExtraction,
            output_config={"effort": "low"},  # short extraction task
            # If the model declines, the API retries on a fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable(f"connection error: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("rate limited") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.AnthropicError as exc:
        raise LLMUnavailable(str(exc)) from exc
    except TypeError as exc:
        # The SDK reports missing credentials as a TypeError when the request is built.
        # Any other TypeError is a programming error and must not be hidden.
        if "authentication" not in str(exc).lower():
            raise
        raise LLMUnavailable("no Anthropic credentials configured") from exc

    if response.stop_reason == "refusal":
        raise LLMUnavailable("model declined the request")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise LLMUnavailable(f"no usable structured output (stop_reason={response.stop_reason})")

    usage = LLMUsage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )
    return response.parsed_output, usage


TranslatorMode = Literal["rules", "llm", "auto"]


def translate_alert(
    text: str,
    capacity_kwh: float,
    horizon_hours: int = 24,
    horizon_start: str | None = None,
    mode: TranslatorMode = "rules",
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[ScheduleOverridePatch, str]:
    """Translate an alert into overrides. Returns (patch, source) with source "llm" or "rules".

    mode="rules": deterministic regex parser only (free, offline).
    mode="llm":   Claude only; raises LLMUnavailable if it cannot be used.
    mode="auto":  Claude, falling back to the rule parser on any failure.
    """
    if mode == "rules":
        return translate_exception_text(text, capacity_kwh=capacity_kwh), "rules"
    try:
        extraction, usage = extract_with_claude(text, capacity_kwh, horizon_hours, horizon_start, client, model)
    except LLMUnavailable as exc:
        if mode == "llm":
            raise
        logger.warning("LLM translator unavailable, using rule parser", extra={"reason": str(exc)})
        patch = translate_exception_text(text, capacity_kwh=capacity_kwh)
        patch.notes.append(f"LLM translator unavailable ({exc}); used rule parser")
        return patch, "rules"
    logger.info("alert extracted by LLM", extra={"model": model, **usage.__dict__})
    return extraction_to_patch(extraction, capacity_kwh, horizon_hours), "llm"
