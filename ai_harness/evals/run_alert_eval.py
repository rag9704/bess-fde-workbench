"""Score the alert translator against a labelled set of operator alerts (English + Japanese).

    python -m ai_harness.evals.run_alert_eval --mode rules          # free, offline
    python -m ai_harness.evals.run_alert_eval --mode llm            # calls Claude (~30 requests)
    python -m ai_harness.evals.run_alert_eval --mode llm --out ai_harness/evals/results_llm.json

A case passes only if *all* overrides match: the reserve floors, the import caps and the
end-of-horizon SoC (values within 1 kWh / 1 kW). Every case is a 24 h horizon starting 00:00
on a 2,000 kWh battery.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time

from ai_harness.exception_translator import ScheduleOverridePatch, translate_exception_text
from ai_harness.llm_translator import DEFAULT_MODEL, LLMUnavailable, extract_with_claude, extraction_to_patch

CASES = Path(__file__).with_name("alert_cases.jsonl")
CAPACITY_KWH = 2000.0
HORIZON = 24
# Claude Opus 5.5 list prices, USD per million tokens (input, output)
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0)}


@dataclass
class CaseResult:
    id: str
    passed: bool
    fields: dict[str, bool]
    got: dict
    expected: dict
    error: str | None = None
    tags: str = ""


@dataclass
class EvalReport:
    mode: str
    results: list[CaseResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _patch_dict(p: ScheduleOverridePatch) -> dict:
    return {
        "soc_min": [list(x) for x in p.soc_min_overrides],
        "grid_cap": [list(x) for x in p.grid_cap_overrides],
        "end_soc_min_kwh": p.end_soc_min_kwh,
    }


def _same_overrides(got: list, expected: list, tol: float = 1.0) -> bool:
    def key(x):
        return (int(x[0]), int(x[1]))

    g = sorted(got, key=key)
    e = sorted(expected, key=key)
    return len(g) == len(e) and all(key(a) == key(b) and abs(a[2] - b[2]) <= tol for a, b in zip(g, e, strict=True))


def score(got: dict, expected: dict) -> dict[str, bool]:
    ge, ee = got["end_soc_min_kwh"], expected["end_soc_min_kwh"]
    return {
        "soc_min": _same_overrides(got["soc_min"], expected["soc_min"]),
        "grid_cap": _same_overrides(got["grid_cap"], expected["grid_cap"]),
        "end_soc_min_kwh": (ge is None and ee is None) or (ge is not None and ee is not None and abs(ge - ee) <= 1.0),
    }


def run(mode: str, cases: list[dict], model: str = DEFAULT_MODEL, client=None) -> EvalReport:
    report = EvalReport(mode=mode)
    t0 = time.perf_counter()
    for case in cases:
        error = None
        try:
            if mode == "rules":
                patch = translate_exception_text(case["text"], capacity_kwh=CAPACITY_KWH)
            else:
                extraction, usage = extract_with_claude(
                    case["text"], CAPACITY_KWH, HORIZON, "2026-04-01 00:00", client=client, model=model
                )
                report.input_tokens += usage.input_tokens
                report.output_tokens += usage.output_tokens
                patch = extraction_to_patch(extraction, CAPACITY_KWH, HORIZON)
            got = _patch_dict(patch)
        except LLMUnavailable as exc:
            if not report.results:  # first call failed: no point trying the rest
                raise
            got, error = {"soc_min": [], "grid_cap": [], "end_soc_min_kwh": None}, str(exc)
        fields = score(got, case["expected"])
        report.results.append(
            CaseResult(
                case["id"],
                all(fields.values()) and error is None,
                fields,
                got,
                case["expected"],
                error,
                case.get("tags", ""),
            )
        )
    report.seconds = time.perf_counter() - t0
    return report


def format_report(report: EvalReport, model: str = DEFAULT_MODEL) -> str:
    n = len(report.results)
    lines = [
        f"Mode: {report.mode}   cases: {n}   passed: {sum(r.passed for r in report.results)}   "
        f"accuracy: {report.accuracy:.0%}   time: {report.seconds:.1f}s"
    ]
    field_ok = Counter()
    for r in report.results:
        field_ok.update(k for k, v in r.fields.items() if v)
    lines.append(
        "Per field: " + ", ".join(f"{k} {field_ok[k]}/{n}" for k in ("soc_min", "grid_cap", "end_soc_min_kwh"))
    )
    if report.mode == "llm":
        pin, pout = PRICE_PER_MTOK.get(model, (0.0, 0.0))
        cost = report.input_tokens / 1e6 * pin + report.output_tokens / 1e6 * pout
        lines.append(f"Tokens: {report.input_tokens:,} in / {report.output_tokens:,} out   est. cost: ${cost:.3f}")
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append("\nFailures:")
        for r in failed:
            bad = [k for k, v in r.fields.items() if not v]
            lines.append(f"  - {r.id} [{r.tags}] wrong: {', '.join(bad) or 'error'}")
            for k in bad:
                lines.append(f"      {k}: got {r.got[k]}  expected {r.expected[k]}")
            if r.error:
                lines.append(f"      error: {r.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("rules", "llm"), default="rules")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--out", type=Path, default=None, help="Write per-case results as JSON")
    args = ap.parse_args(argv)

    try:
        report = run(args.mode, load_cases(args.cases), model=args.model)
    except LLMUnavailable as exc:
        print(f"Cannot run the LLM eval: {exc}", file=sys.stderr)
        return 3
    print(format_report(report, args.model))
    if args.out:
        args.out.write_text(
            json.dumps(
                {"mode": report.mode, "accuracy": report.accuracy, "results": [r.__dict__ for r in report.results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
