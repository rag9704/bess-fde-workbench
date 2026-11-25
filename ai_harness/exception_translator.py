from dataclasses import dataclass, field
import re
from typing import Any
import numpy as np


@dataclass
class ScheduleOverridePatch:
    """Structured operational overrides derived from natural-language events."""

    soc_min_overrides: list[tuple[int, int, float]] = field(default_factory=list)
    grid_cap_overrides: list[tuple[int, int, float]] = field(default_factory=list)
    end_soc_min_kwh: float | None = None
    notes: list[str] = field(default_factory=list)

    def apply(
        self,
        horizon_len: int,
        default_soc_min_kwh: float,
        default_grid_cap_kw: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (soc_min_arr of length T+1, grid_cap_arr of length T)."""
        soc_min = np.full(horizon_len + 1, default_soc_min_kwh, dtype=float)
        grid_cap = np.full(horizon_len, default_grid_cap_kw, dtype=float)

        for start_h, end_h, val in self.soc_min_overrides:
            lo = max(0, start_h)
            hi = min(horizon_len + 1, end_h + 1)
            if lo < hi:
                soc_min[lo:hi] = np.maximum(soc_min[lo:hi], val)

        for start_h, end_h, val in self.grid_cap_overrides:
            lo = max(0, start_h)
            hi = min(horizon_len, end_h + 1)
            if lo < hi:
                grid_cap[lo:hi] = np.minimum(grid_cap[lo:hi], val)

        if self.end_soc_min_kwh is not None:
            soc_min[horizon_len] = max(soc_min[horizon_len], self.end_soc_min_kwh)

        return soc_min, grid_cap

    def diff_summary(self) -> list[str]:
        lines: list[str] = []
        for s, e, v in self.soc_min_overrides:
            lines.append(f"SoC reserve floor >= {v:.1f} kWh for hour indices [{s}..{e}]")
        for s, e, v in self.grid_cap_overrides:
            lines.append(f"Grid import cap <= {v:.1f} kW for hour indices [{s}..{e}]")
        if self.end_soc_min_kwh is not None:
            lines.append(f"Terminal SoC floor >= {self.end_soc_min_kwh:.1f} kWh")
        lines.extend(self.notes)
        return lines or ["No overrides extracted"]


def translate_exception_text(
    text: str,
    capacity_kwh: float = 2000.0,
    llm_callable: Any | None = None,
) -> ScheduleOverridePatch:
    """Translate an operator alert into a structured override patch.

    Uses deterministic regex/keyword parsing by default; if `llm_callable(prompt: str) -> str`
    is supplied, it can augment or refine edge cases while keeping output typed and bounded.
    """
    _ = llm_callable  # Hook for optional LLM client integration
    patch = ScheduleOverridePatch()
    lower = text.lower()

    # Parse optional hour window like "hours 12-18" or "from t=10 to 20" or "13:00 to 17:00"
    start_h, end_h = 0, 23
    win_match = re.search(r"(?:hours?|t\s*=?\s*|from\s+)(\d{1,2})(?::00)?\s*(?:-|to|~)\s*(\d{1,2})(?::00)?", lower)
    if win_match:
        start_h = int(win_match.group(1))
        end_h = int(win_match.group(2))

    # Reserve / SoC percentage or kWh
    pct_match = re.search(r"(?:reserve|soc)[^0-9]{0,20}(\d{1,3})\s*%", lower)
    kwh_match = re.search(r"(?:reserve|soc)[^0-9]{0,20}(\d{2,5})\s*kwh", lower)
    if pct_match:
        pct = min(max(float(pct_match.group(1)), 0.0), 100.0)
        kwh_val = round(capacity_kwh * (pct / 100.0), 1)
        patch.soc_min_overrides.append((start_h, end_h, kwh_val))
    elif kwh_match:
        patch.soc_min_overrides.append((start_h, end_h, float(kwh_match.group(1))))
    elif any(k in lower for k in ["typhoon", "storm", "台風", "警報", "outage", "停電"]):
        # Safe storm/outage default: 75% reserve over window
        patch.soc_min_overrides.append((start_h, end_h, round(0.75 * capacity_kwh, 1)))
        patch.notes.append("Applied 75% storm/outage reserve heuristic")

    # Grid import cap in kW (e.g. "cap grid import at 650 kw", "transformer limit 500kw")
    cap_match = re.search(r"(?:cap|limit|import|transformer|受電)[^0-9]{0,25}(\d{2,5})\s*kw", lower)
    if cap_match:
        patch.grid_cap_overrides.append((start_h, end_h, float(cap_match.group(1))))

    return patch