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
import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .exception_translator import ScheduleOverridePatch, translate_exception_text
from .llm_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    LLMConfig,
    LLMUnavailable,
    openai_client,
    resolve_llm_config,
    status_error_message,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL  # used when the Anthropic backend is picked and no model is set
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
    """Deterministic, bounded conversion from the LLM's extraction to model overrides."""
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


# --------------------------------------------------------------------------- LLM calls

__all__ = ["LLMUnavailable", "LLMConfig", "translate_alert", "extract_alert", "extraction_to_patch"]


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


def _user_message(text: str, capacity_kwh: float, horizon_hours: int, horizon_start: str | None) -> str:
    return (
        f"Horizon: {horizon_hours} hourly steps starting {horizon_start or '00:00 (hour index = clock hour)'}.\n"
        f"Battery capacity: {capacity_kwh:.0f} kWh.\n\n"
        f"Operator alert:\n<alert>\n{text}\n</alert>"
    )


def extract_alert(
    text: str,
    capacity_kwh: float,
    horizon_hours: int,
    horizon_start: str | None = None,
    client: Any | None = None,
    config: LLMConfig | None = None,
) -> tuple[AlertExtraction, LLMUsage]:
    """Ask the configured LLM for a structured extraction. Raises LLMUnavailable on any failure."""
    cfg = config or resolve_llm_config()
    user = _user_message(text, capacity_kwh, horizon_hours, horizon_start)
    if cfg.provider == "openai":
        return _extract_openai(user, cfg, client)
    return _extract_anthropic(user, cfg, client)


def _extract_anthropic(user: str, cfg: LLMConfig, client: Any | None) -> tuple[AlertExtraction, LLMUsage]:
    import anthropic

    try:
        client = client or anthropic.Anthropic()
    except anthropic.AnthropicError as exc:  # e.g. no credentials configured
        raise LLMUnavailable(str(exc)) from exc
    try:
        response = client.beta.messages.parse(
            model=cfg.model,
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
        raise LLMUnavailable(status_error_message(exc, cfg)) from exc
    except anthropic.AnthropicError as exc:
        raise LLMUnavailable(str(exc)) from exc
    except TypeError as exc:
        # The SDK reports missing credentials as a TypeError when the request is built.
        # Any other TypeError is a programming error and must not be hidden.
        if "authentication" not in str(exc).lower():
            raise
        raise LLMUnavailable("no Anthropic credentials configured (set ANTHROPIC_API_KEY)") from exc

    if response.stop_reason == "refusal":
        raise LLMUnavailable("model declined the request")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise LLMUnavailable(f"no usable structured output (stop_reason={response.stop_reason})")
    usage = LLMUsage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )
    return response.parsed_output, usage


def _inline_refs(schema: dict) -> dict:
    """Pydantic emits $defs/$ref; some OpenAI-compatible servers only accept a self-contained schema."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


ALERT_SCHEMA = _inline_refs(AlertExtraction.model_json_schema())
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _extract_openai(user: str, cfg: LLMConfig, client: Any | None) -> tuple[AlertExtraction, LLMUsage]:
    import openai

    client = client or openai_client(cfg)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    json_schema = {"type": "json_schema", "json_schema": {"name": "alert_extraction", "schema": ALERT_SCHEMA}}
    try:
        try:
            response = client.chat.completions.create(model=cfg.model, messages=messages, response_format=json_schema)
        except openai.BadRequestError as exc:
            if "response_format" not in str(exc) and "json_schema" not in str(exc):
                raise
            # Endpoint without JSON-schema support: ask for plain JSON and put the schema in the prompt.
            logger.info("endpoint rejected json_schema; retrying with json_object", extra={"model": cfg.model})
            messages[0] = {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n\nReply with one JSON object matching this JSON Schema:\n"
                + json.dumps(ALERT_SCHEMA),
            }
            response = client.chat.completions.create(
                model=cfg.model, messages=messages, response_format={"type": "json_object"}
            )
    except openai.APIConnectionError as exc:
        raise LLMUnavailable(f"connection error: {exc}") from exc
    except openai.RateLimitError as exc:
        raise LLMUnavailable("rate limited") from exc
    except openai.APIStatusError as exc:
        raise LLMUnavailable(status_error_message(exc, cfg)) from exc
    except openai.OpenAIError as exc:
        raise LLMUnavailable(str(exc)) from exc

    if not response.choices:
        raise LLMUnavailable("empty response")
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise LLMUnavailable("model declined the request")
    if choice.finish_reason == "length":
        raise LLMUnavailable("no usable structured output (finish_reason=length)")
    content = _FENCE.sub("", (choice.message.content or "").strip())
    try:
        extraction = AlertExtraction.model_validate_json(content)
    except ValidationError as exc:
        raise LLMUnavailable(f"output does not match the schema: {exc.error_count()} error(s)") from exc
    usage = LLMUsage(
        input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
    )
    return extraction, usage


def extract_with_claude(*args, **kwargs) -> tuple[AlertExtraction, LLMUsage]:
    """Backward-compatible name for `extract_alert`."""
    return extract_alert(*args, **kwargs)


TranslatorMode = Literal["rules", "llm", "auto"]


def translate_alert(
    text: str,
    capacity_kwh: float,
    horizon_hours: int = 24,
    horizon_start: str | None = None,
    mode: TranslatorMode = "rules",
    client: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[ScheduleOverridePatch, str]:
    """Translate an alert into overrides. Returns (patch, source) with source "llm" or "rules".

    mode="rules": deterministic regex parser only (free, offline).
    mode="llm":   LLM only; raises LLMUnavailable if it cannot be used.
    mode="auto":  LLM, falling back to the rule parser on any failure.
    provider/model override the environment (see ai_harness.llm_provider).
    """
    if mode == "rules":
        return translate_exception_text(text, capacity_kwh=capacity_kwh), "rules"
    try:
        cfg = resolve_llm_config(provider, model)
        extraction, usage = extract_alert(text, capacity_kwh, horizon_hours, horizon_start, client, cfg)
    except LLMUnavailable as exc:
        if mode == "llm":
            raise
        logger.warning("LLM translator unavailable, using rule parser", extra={"reason": str(exc)})
        patch = translate_exception_text(text, capacity_kwh=capacity_kwh)
        patch.notes.append(f"LLM translator unavailable ({exc}); used rule parser")
        return patch, "rules"
    logger.info("alert extracted by LLM", extra={"llm": cfg.label, **usage.__dict__})
    return extraction_to_patch(extraction, capacity_kwh, horizon_hours), "llm"
