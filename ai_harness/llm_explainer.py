"""Claude-written explanation of an infeasible scenario, for the site operator.

The solver's elastic recovery already says *which* limit breaks, *when*, and *by how much*.
This turns that into a short, actionable note: why the limits conflict and which decision
the operator has to make. The model only explains; it never changes the schedule.
"""

import logging
from typing import Any

from core_model.formulation import BessParams

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"

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
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Return a short explanation, or None if Claude is unavailable (the brief still has the facts)."""
    import anthropic

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
        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except (anthropic.AnthropicError, TypeError) as exc:
        # TypeError: the SDK's "no credentials" error; anything else is re-raised below.
        if isinstance(exc, TypeError) and "authentication" not in str(exc).lower():
            raise
        logger.warning("LLM explanation unavailable", extra={"reason": str(exc)})
        return None

    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None
