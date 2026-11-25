from core_model.solver_runner import DispatchResult


def build_operator_brief(
    strict_result: DispatchResult,
    recovery_result: DispatchResult | None = None,
    timestamps: list[str] | None = None,
) -> str:
    """Produce a concise engineering + operations summary from solver & slack results."""
    lines: list[str] = []
    if strict_result.usable and strict_result.objective_yen is not None:
        gap = ""
        if strict_result.status == "FEASIBLE" and strict_result.mip_gap is not None:
            gap = f" — stopped at limit, within {strict_result.mip_gap:.2%} of optimal"
        lines.append(
            f"Status: {strict_result.status} ({strict_result.backend}, {strict_result.solve_time_sec:.3f}s){gap} | "
            f"Objective: ¥{strict_result.objective_yen:,.0f} | Peak Import: {strict_result.peak_import_kw:.1f} kW"
        )
        lines.append(
            f"Energy: Import {strict_result.total_import_kwh:,.0f} kWh, "
            f"Export {strict_result.total_export_kwh:,.0f} kWh, "
            f"Charge {strict_result.total_charge_kwh:,.0f} kWh, "
            f"Discharge {strict_result.total_discharge_kwh:,.0f} kWh"
        )
        return "\n".join(lines)

    if strict_result.status == "NO_SOLUTION":
        lines.append(
            f"Status: NO_SOLUTION ({strict_result.solve_time_sec:.3f}s) — the solver hit its time limit before "
            "finding any feasible schedule; raise the time limit or MIP gap."
        )
        return "\n".join(lines)

    lines.append(
        f"Status: STRICT {strict_result.status} ({strict_result.solve_time_sec:.3f}s) — "
        "hard constraints conflicted under current inputs."
    )
    if recovery_result is None or not recovery_result.usable:
        lines.append("Elastic fallback did not find a feasible solution; inspect bounds and load balance.")
        return "\n".join(lines)

    violations = recovery_result.infeasibility_summary(timestamps=timestamps)
    # The strict problem has no solution, so there is no strict cost to compare against;
    # report the recovery schedule's operational cost (slack penalties excluded).
    cost = f"¥{recovery_result.objective_yen:,.0f}" if recovery_result.objective_yen is not None else "N/A"
    lines.append(
        f"Elastic Recovery: {recovery_result.status} ({recovery_result.solve_time_sec:.3f}s) | "
        f"Operational cost (excl. penalties): {cost} | Peak Import: {recovery_result.peak_import_kw:.1f} kW"
    )
    if violations:
        lines.append("Detected constraint conflicts (top items):")
        for item in violations[:8]:
            lines.append(f"  - {item}")
        if len(violations) > 8:
            lines.append(f"  - ...and {len(violations) - 8} more interval violations")
    return "\n".join(lines)
