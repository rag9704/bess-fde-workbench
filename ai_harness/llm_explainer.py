"""LLM-written explanation of an infeasible scenario, for the site operator.

The solver's elastic recovery already says *which* limit breaks, *when*, and *by how much*.
This turns that into a short, actionable note: why the limits conflict and which decision
the operator has to make. The model only explains; it never changes the schedule.
"""

import logging
from typing import Any

from core_model.formulation import BessParams

from .llm_provider import LLMUnavailable, openai_client, resolve_llm_config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an energy-operations engineer explaining an optimizer result to the operator of a commercial
solar + battery site in Japan. The optimizer could not satisfy all hard limits, so it computed the
least-bad schedule and reported where each limit is violated.

Write for a busy operator, in plain language, at most 120 words:
1. One sentence: why the limits conflict (use the numbers given; do the arithmetic if it helps).
2. The decision the operator must make, as 2-3 concrete options with their trade-off.
Do not invent data that is not given. Do not use markdown headings."""


def explain_conflicts(
    alert: str | None,
    overrides: list[str],
    violations: list[str],
    params: BessParams,
    client: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str | None:
    """Return a short explanation, or None if the LLM is unavailable (the brief still has the facts)."""
    shown = violations[:12]
    more = f"\n(... and {len(violations) - len(shown)} more)" if len(violations) > len(shown) else ""
    user = (
        f"Battery: {params.capacity_kwh:.0f} kWh, {params.max_charge_kw:.0f} kW charge / "
        f"{params.max_discharge_kw:.0f} kW discharge. Normal grid import cap {params.grid_import_cap_kw:.0f} kW.\n"
        f"Operator alert: {alert or '(none)'}\n"
        "Overrides applied:\n" + "\n".join(f"- {o}" for o in overrides or ["(none)"]) + "\n"
        "Violations in the least-bad schedule:\n" + "\n".join(f"- {v}" for v in shown) + more
    )
    try:
        cfg = resolve_llm_config(provider, model)
        if cfg.provider == "openai":
            return _explain_openai(user, cfg, client)
        return _explain_anthropic(user, cfg, client)
    except LLMUnavailable as exc:
        logger.warning("LLM explanation unavailable", extra={"reason": str(exc)})
        return None


def _explain_anthropic(user: str, cfg, client: Any | None) -> str | None:
    import anthropic

    try:
        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=cfg.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except (anthropic.AnthropicError, TypeError) as exc:
        # TypeError: the SDK's "no credentials" error; anything else is re-raised.
        if isinstance(exc, TypeError) and "authentication" not in str(exc).lower():
            raise
        raise LLMUnavailable(str(exc)) from exc
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


def _explain_openai(user: str, cfg, client: Any | None) -> str | None:
    import openai

    client = client or openai_client(cfg)
    try:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        )
    except openai.OpenAIError as exc:
        raise LLMUnavailable(str(exc)) from exc
    if not response.choices or getattr(response.choices[0].message, "refusal", None):
        return None
    return (response.choices[0].message.content or "").strip() or None
