"""Provider selection and the OpenAI-compatible backend, tested with a fake client (no network, no key)."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from ai_harness.evals.run_alert_eval import load_cases, run
from ai_harness.llm_explainer import explain_conflicts
from ai_harness.llm_provider import GEMINI_OPENAI_BASE_URL, LLMUnavailable, resolve_llm_config
from ai_harness.llm_translator import ALERT_SCHEMA, translate_alert
from core_model.formulation import BessParams

CAP = 2000.0
CAP_ALERT = {
    "reserves": [],
    "import_caps": [{"start_hour": 13, "end_hour": 17, "cap_kw": 400}],
    "end_soc_min_kwh": None,
    "end_soc_min_percent": None,
    "storm_or_outage_risk": False,
    "storm_window": None,
    "not_applied": [],
}


class FakeCompletions:
    def __init__(self, content=None, finish_reason="stop", exc=None, reject_schema=False, refusal=None):
        self.content = json.dumps(CAP_ALERT) if content is None else content
        self.finish_reason, self.exc, self.reject_schema, self.refusal = finish_reason, exc, reject_schema, refusal
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        if self.reject_schema and kw.get("response_format", {}).get("type") == "json_schema":
            raise openai.BadRequestError("response_format json_schema not supported", response=_resp(400), body=None)
        msg = SimpleNamespace(content=self.content, refusal=self.refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=self.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=80),
        )


def _resp(status: int, text: str = "") -> httpx.Response:
    req = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return httpx.Response(status, text=text, request=req)


def fake_openai(**kw):
    completions = FakeCompletions(**kw)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


# --------------------------------------------------------------------------- picking the backend


def test_defaults_to_claude():
    cfg = resolve_llm_config()
    assert (cfg.provider, cfg.model) == ("anthropic", "claude-opus-5-5")


def test_openai_compatible_key_alone_selects_that_backend(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-key-123")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    cfg = resolve_llm_config()
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.key_var) == (
        "openai",
        "my-model",
        "https://llm.example/v1",
        "LLM_API_KEY",
    )
    assert "secret-key-123" not in repr(cfg)  # the key never appears in logs or reprs


def test_gemini_key_implies_google_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = resolve_llm_config(model="gemini-2.5-flash")
    assert cfg.provider == "openai" and cfg.base_url == GEMINI_OPENAI_BASE_URL


def test_project_llm_settings_beat_a_global_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("LLM_API_KEY", "g")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    cfg = resolve_llm_config()
    assert (cfg.provider, cfg.model) == ("openai", "gemini-2.5-flash")
    assert resolve_llm_config(provider="anthropic").provider == "anthropic"


def test_both_keys_prefer_claude_unless_told(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolve_llm_config().provider == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert resolve_llm_config().provider == "openai"
    assert resolve_llm_config(provider="anthropic").provider == "anthropic"  # explicit argument wins


def test_openai_compatible_url_in_anthropic_base_url_is_explained(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    with pytest.raises(LLMUnavailable, match="OpenAI-compatible"):
        resolve_llm_config()


def test_openai_backend_needs_a_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    with pytest.raises(LLMUnavailable, match="LLM_MODEL"):
        translate_alert("cap 400 kW", CAP, mode="llm")
    patch, source = translate_alert("reserve 50%", CAP, mode="auto")
    assert source == "rules" and any("LLM_MODEL" in n for n in patch.notes)


# --------------------------------------------------------------------------- OpenAI-compatible backend


def test_openai_request_shape_and_result():
    client, completions = fake_openai()
    patch, source = translate_alert(
        "Cap import at 400 kW 13-17", CAP, mode="llm", client=client, provider="openai", model="gemini-2.5-flash"
    )
    assert source == "llm" and patch.grid_cap_overrides == [(13, 17, 400.0)]
    call = completions.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["messages"][0]["role"] == "system" and "<alert>" in call["messages"][1]["content"]
    assert call["response_format"]["json_schema"]["schema"] == ALERT_SCHEMA


def test_schema_is_self_contained():
    assert "$ref" not in json.dumps(ALERT_SCHEMA) and "$defs" not in ALERT_SCHEMA


def test_falls_back_to_json_object_when_schema_unsupported():
    client, completions = fake_openai(reject_schema=True)
    patch, _ = translate_alert("cap", CAP, mode="llm", client=client, provider="openai", model="m")
    assert [c["response_format"]["type"] for c in completions.calls] == ["json_schema", "json_object"]
    assert "JSON Schema" in completions.calls[1]["messages"][0]["content"]
    assert patch.grid_cap_overrides == [(13, 17, 400.0)]


def test_code_fenced_json_is_accepted():
    client, _ = fake_openai(content="```json\n" + json.dumps(CAP_ALERT) + "\n```")
    patch, _ = translate_alert("cap", CAP, mode="llm", client=client, provider="openai", model="m")
    assert patch.grid_cap_overrides == [(13, 17, 400.0)]


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"content": '{"reserves": "oops"}'}, "schema"),
        ({"content": "not json"}, "schema"),
        ({"finish_reason": "length"}, "length"),
        ({"refusal": "no"}, "declined"),
        ({"exc": openai.NotFoundError("nf", response=_resp(404, "404 page not found"), body=None)}, "404"),
    ],
)
def test_unusable_openai_responses_fall_back(kw, match):
    client, _ = fake_openai(**kw)
    with pytest.raises(LLMUnavailable, match=match):
        translate_alert("reserve 50%", CAP, mode="llm", client=client, provider="openai", model="m")
    patch, source = translate_alert("reserve 50%", CAP, mode="auto", client=client, provider="openai", model="m")
    assert source == "rules" and patch.soc_min_overrides


def test_404_message_names_url_and_body():
    client, _ = fake_openai(exc=openai.NotFoundError("nf", response=_resp(404, "404 page not found"), body=None))
    with pytest.raises(LLMUnavailable) as info:
        translate_alert("x", CAP, mode="llm", client=client, provider="openai", model="m")
    msg = str(info.value)
    assert "https://example.test/v1/chat/completions" in msg and "404 page not found" in msg and "'m'" in msg


def test_openai_explainer():
    client, completions = fake_openai(content="Cap and reserve conflict at 14:00.")
    text = explain_conflicts("x", [], ["v"], BessParams(), client=client, provider="openai", model="m")
    assert text == "Cap and reserve conflict at 14:00." and "response_format" not in completions.calls[0]


def test_eval_runs_on_openai_backend():
    client, _ = fake_openai()
    cases = [c for c in load_cases() if c["expected"]["grid_cap"] == [[13, 17, 400]]][:1] or load_cases()[:1]
    report = run("llm", cases, config=resolve_llm_config("openai", "m"), client=client)
    assert report.llm.startswith("openai:m") and report.input_tokens == 300
