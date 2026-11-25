"""Claude-backed alert translation and explanation, tested with a fake client (no network, no key)."""

from types import SimpleNamespace

import anthropic
import pytest

from ai_harness.evals.run_alert_eval import load_cases, run, score
from ai_harness.llm_explainer import explain_conflicts
from ai_harness.llm_translator import (
    AlertExtraction,
    ImportCapRequest,
    LLMUnavailable,
    ReserveRequest,
    extraction_to_patch,
    translate_alert,
)
from core_model.formulation import BessParams
from service.pipeline import run_scenario

CAP = 2000.0


def extraction(**kw) -> AlertExtraction:
    base = dict(
        reserves=[],
        import_caps=[],
        end_soc_min_kwh=None,
        end_soc_min_percent=None,
        storm_or_outage_risk=False,
        storm_window=None,
        not_applied=[],
    )
    base.update(kw)
    return AlertExtraction(**base)


class FakeMessages:
    def __init__(self, parsed=None, stop_reason="end_turn", exc=None, text="Explanation."):
        self.parsed, self.stop_reason, self.exc, self.text = parsed, stop_reason, exc, text
        self.calls: list[dict] = []

    def _usage(self):
        return SimpleNamespace(input_tokens=120, output_tokens=40)

    def parse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason, usage=self._usage())

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)], stop_reason=self.stop_reason, usage=self._usage()
        )


def fake_client(**kw):
    messages = FakeMessages(**kw)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


# --------------------------------------------------------------------------- guardrails


def test_guardrails_clamp_and_bound_everything():
    ex = extraction(
        reserves=[
            ReserveRequest(start_hour=20, end_hour=30, unit="kwh", value=5000),  # over capacity, past horizon
            ReserveRequest(start_hour=5, end_hour=2, unit="percent", value=150),  # reversed window, >100 %
            ReserveRequest(start_hour=40, end_hour=50, unit="percent", value=50),  # entirely outside horizon
        ],
        import_caps=[
            ImportCapRequest(start_hour=0, end_hour=23, cap_kw=-10),  # negative: rejected
            ImportCapRequest(start_hour=0, end_hour=23, cap_kw=400),
        ],
        end_soc_min_percent=250,
    )
    patch = extraction_to_patch(ex, CAP, horizon_hours=24)
    assert patch.soc_min_overrides == [(20, 23, 2000.0), (2, 5, 2000.0)]
    assert patch.grid_cap_overrides == [(0, 23, 400.0)]
    assert patch.end_soc_min_kwh == 2000.0
    notes = " ".join(patch.notes)
    assert "clamped" in notes and "outside the horizon" in notes and "negative" in notes


def test_storm_policy_is_applied_by_code_not_model():
    patch = extraction_to_patch(
        extraction(storm_or_outage_risk=True, storm_window=ImportCapRequest(start_hour=15, end_hour=22, cap_kw=0)),
        CAP,
        24,
    )
    assert patch.soc_min_overrides == [(15, 22, 1500.0)]
    # An explicit reserve wins over the storm default
    explicit = extraction(
        storm_or_outage_risk=True, reserves=[ReserveRequest(start_hour=0, end_hour=23, unit="percent", value=60)]
    )
    assert extraction_to_patch(explicit, CAP, 24).soc_min_overrides == [(0, 23, 1200.0)]


def test_unmapped_requests_are_surfaced_not_dropped():
    patch = extraction_to_patch(extraction(not_applied=["raise export price to 12 yen"]), CAP, 24)
    assert patch.soc_min_overrides == [] and any("export price" in n for n in patch.notes)


# --------------------------------------------------------------------------- the Claude call


def test_llm_request_shape_and_result():
    client, msgs = fake_client(
        parsed=extraction(import_caps=[ImportCapRequest(start_hour=13, end_hour=17, cap_kw=400)])
    )
    patch, source = translate_alert(
        "Grid maintenance 13:00 to 17:00: limit import to 400 kW",
        CAP,
        24,
        "2026-04-01 00:00",
        mode="llm",
        client=client,
    )
    assert source == "llm" and patch.grid_cap_overrides == [(13, 17, 400.0)]
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is AlertExtraction
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    assert "<alert>\nGrid maintenance" in call["messages"][0]["content"]
    assert "2026-04-01 00:00" in call["messages"][0]["content"]


def test_auto_falls_back_to_rules_on_api_error():
    client, _ = fake_client(exc=anthropic.AnthropicError("boom"))
    patch, source = translate_alert("reserve 80% from 14 to 20", CAP, mode="auto", client=client)
    assert source == "rules" and patch.soc_min_overrides == [(14, 20, 1600.0)]
    assert any("used rule parser" in n for n in patch.notes)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses(stop_reason):
    client, _ = fake_client(parsed=extraction(), stop_reason=stop_reason)
    with pytest.raises(LLMUnavailable):
        translate_alert("reserve 50%", CAP, mode="llm", client=client)
    patch, source = translate_alert("reserve 50%", CAP, mode="auto", client=client)
    assert source == "rules" and patch.soc_min_overrides == [(0, 23, 1000.0)]


def test_programming_errors_are_not_swallowed():
    client, _ = fake_client(exc=TypeError("unexpected keyword argument 'foo'"))
    with pytest.raises(TypeError):
        translate_alert("reserve 50%", CAP, mode="auto", client=client)


def test_rules_mode_never_calls_the_api():
    client, msgs = fake_client(parsed=extraction())
    translate_alert("reserve 50%", CAP, mode="rules", client=client)
    assert msgs.calls == []


# --------------------------------------------------------------------------- pipeline integration


def test_pipeline_uses_llm_overrides_and_only_tightens(day):
    params = BessParams()
    client, _ = fake_client(
        parsed=extraction(
            reserves=[ReserveRequest(start_hour=14, end_hour=20, unit="percent", value=80)],
            end_soc_min_kwh=300.0,  # lower than the site minimum (800): must NOT loosen it
        )
    )
    result = run_scenario(day, params, alert="whatever", translator="llm", llm_client=client, compare=False)
    assert result.translator_source == "llm"
    assert result.params.end_soc_min_kwh == params.end_soc_min_kwh
    assert result.strict.usable
    assert result.strict.soc_kwh[14:21].min() >= 1600.0 - 1e-5


def test_explainer_runs_only_on_conflicts(day, params):
    client, msgs = fake_client(text="Import cap too low for the load; raise the cap or accept shedding.")
    ok = run_scenario(day, params, explain=True, llm_client=client, compare=False)
    assert ok.explanation is None and msgs.calls == []

    result = run_scenario(
        day, params, alert="Typhoon warning, cap grid import at 300 kW", explain=True, llm_client=client
    )
    assert result.recovery is not None and "raise the cap" in result.explanation
    assert "Grid import cap exceeded" in msgs.calls[0]["messages"][0]["content"]


def test_explainer_degrades_gracefully():
    client, _ = fake_client(exc=anthropic.AnthropicError("down"))
    assert explain_conflicts("x", [], ["v"], BessParams(), client=client) is None


# --------------------------------------------------------------------------- eval set


def test_eval_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 30 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        e = c["expected"]
        for s, t, v in e["soc_min"]:
            assert 0 <= s <= t <= 23 and 0 <= v <= CAP
        for s, t, v in e["grid_cap"]:
            assert 0 <= s <= t <= 23 and v >= 0


def test_eval_labels_match_code_policy():
    """Feeding a perfect extraction through the guardrails must reproduce the eval labels."""
    cases = {c["id"]: c["expected"] for c in load_cases()}
    perfect = {
        "over-capacity-clamp": extraction(reserves=[ReserveRequest(start_hour=0, end_hour=23, unit="kwh", value=2500)]),
        "storm-window": extraction(
            storm_or_outage_risk=True, storm_window=ImportCapRequest(start_hour=15, end_hour=22, cap_kw=0)
        ),
        "two-windows": extraction(
            reserves=[ReserveRequest(start_hour=0, end_hour=6, unit="percent", value=30)],
            import_caps=[ImportCapRequest(start_hour=10, end_hour=15, cap_kw=700)],
        ),
        "end-soc-pct": extraction(end_soc_min_percent=70),
    }
    for cid, ex in perfect.items():
        p = extraction_to_patch(ex, CAP, 24)
        got = {
            "soc_min": [list(x) for x in p.soc_min_overrides],
            "grid_cap": [list(x) for x in p.grid_cap_overrides],
            "end_soc_min_kwh": p.end_soc_min_kwh,
        }
        assert all(score(got, cases[cid]).values()), cid


def test_rules_baseline_accuracy_is_tracked():
    report = run("rules", load_cases())
    # The regex parser's known ceiling; raise this number when the rules improve.
    assert report.accuracy == pytest.approx(16 / 30)
