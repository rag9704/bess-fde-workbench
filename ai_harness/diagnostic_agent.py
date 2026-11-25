from core_model.solver_runner import DispatchResult


def build_operator_brief(
    strict_result: DispatchResult,
    recovery_result: DispatchResult | None = None,
    timestamps: list[str] | None = None,
) -> str:
    """Produce a concise engineering + operations summary from solver & slack results."""
    lines: list[str] = []
    if strict_result.status == "OPTIMAL" and strict_result.objective_yen is not None:
        lines.append(
            f"Status: OPTIMAL ({strict_result.backend}, {strict_result.solve_time_sec:.3f}s) | "
            f"Objective: ¥{strict_result.objective_yen:,.0f} | Peak Import: {strict_result.peak_import_kw:.1f} kW"
        )
        lines.append(
            f"Energy: Import {strict_result.total_import_kwh:,.0f} kWh, "
            f"Export {strict_result.total_export_kwh:,.0f} kWh, "
            f"Charge {strict_result.total_charge_kwh:,.0f} kWh, "
            f"Discharge {strict_result.total_discharge_kwh:,.0f} kWh"
        )
        return "\n".join(lines)

    lines.append(
        f"Status: STRICT {strict_result.status} ({strict_result.solve_time_sec:.3f}s) — "
        "hard constraints conflicted under current inputs."
    )
    if recovery_result is None or recovery_result.status != "OPTIMAL":
        lines.append("Elastic fallback did not find a feasible solution; inspect bounds and load balance.")
        return "\n".join(lines)

    violations = recovery_result.infeasibility_summary(timestamps=timestamps)
    cost_delta = (
        f"¥{recovery_result.objective_yen - strict_result.objective_yen:+,.0f}"
        if (recovery_result.objective_yen is not None and strict_result.objective_yen is not None)
        else f"¥{recovery_result.objective_yen:,.0f}" if recovery_result.objective_yen is not None else "N/A"
    )
    lines.append(
        f"Elastic Recovery: OPTIMAL ({recovery_result.solve_time_sec:.3f}s) | "
        f"Recovery Objective: {cost_delta} vs strict | Peak Import: {recovery_result.peak_import_kw:.1f} kW"
    )
    if violations:
        lines.append("Detected constraint conflicts (top items):")
        for item in violations[:8]:
            lines.append(f"  - {item}")
        if len(violations) > 8:
            lines.append(f"  - ...and {len(violations) - 8} more interval violations")
    return "\n".join(lines)