"""Which LLM backend reads alerts: Anthropic (Claude) or any OpenAI-compatible endpoint.

The backend is chosen from the environment, so the same code runs with whichever key you have:

    # Claude (Anthropic API)
    export ANTHROPIC_API_KEY=sk-ant-...            # model: claude-opus-5-5 unless LLM_MODEL is set

    # Any OpenAI-compatible endpoint (OpenAI, Gemini, Azure, vLLM, Ollama, LiteLLM, ...)
    export LLM_API_KEY=...                         # or OPENAI_API_KEY / GEMINI_API_KEY
    export LLM_BASE_URL=https://.../v1             # optional; GEMINI_API_KEY alone implies Google's endpoint
    export LLM_MODEL=gemini-2.5-flash              # required for this backend

    export LLM_PROVIDER=anthropic|openai           # override the choice below

With several keys set: LLM_API_KEY / LLM_BASE_URL select the OpenAI-compatible backend; otherwise an
ANTHROPIC_API_KEY selects Claude.

`--llm-provider` / `--llm-model` on the CLI and the eval override the environment.
"""

from dataclasses import dataclass, field
import os
from typing import Literal

Provider = Literal["anthropic", "openai"]
PROVIDERS: tuple[str, ...] = ("anthropic", "openai")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_KEY_VARS = ("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


class LLMUnavailable(Exception):
    """The LLM could not produce a usable result (no credentials, API error, refusal, bad output, ...)."""


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider
    model: str | None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)  # never logged or printed
    key_var: str | None = None  # name of the env var the key came from

    @property
    def label(self) -> str:
        where = f" @ {self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model}{where}"


def _first_env(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return name, value
    return None, None


def _looks_openai_compatible(url: str) -> bool:
    return "/openai" in url or url.rstrip("/").endswith("/chat/completions")


def resolve_llm_config(provider: str | None = None, model: str | None = None) -> LLMConfig:
    """Pick provider, model, endpoint and key. Raises LLMUnavailable for an unusable setup."""
    provider = provider or os.getenv("LLM_PROVIDER") or None
    if provider is not None and provider not in PROVIDERS:
        raise LLMUnavailable(f"unknown LLM provider {provider!r}; use one of {', '.join(PROVIDERS)}")

    a_var, a_key = _first_env("ANTHROPIC_API_KEY")
    o_var, o_key = _first_env(*OPENAI_KEY_VARS)
    if provider is None:
        # LLM_API_KEY / LLM_BASE_URL are this project's own settings, so they are a deliberate choice;
        # ANTHROPIC_API_KEY is often set globally for other tools. Explicit project settings win.
        explicit_openai = bool(os.getenv("LLM_API_KEY") or os.getenv("LLM_BASE_URL"))
        provider = "openai" if o_key and (explicit_openai or not a_key) else "anthropic"
    model = model or os.getenv("LLM_MODEL") or None

    if provider == "anthropic":
        base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        if base_url and _looks_openai_compatible(base_url):
            raise LLMUnavailable(
                f"ANTHROPIC_BASE_URL={base_url} is an OpenAI-compatible endpoint, but the Anthropic backend sends "
                "Anthropic Messages requests. Unset ANTHROPIC_BASE_URL and use LLM_PROVIDER=openai with "
                "LLM_BASE_URL / LLM_API_KEY / LLM_MODEL instead."
            )
        return LLMConfig(
            "anthropic", model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL, base_url, a_key, a_var
        )

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    if base_url is None and o_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        base_url = GEMINI_OPENAI_BASE_URL
    return LLMConfig("openai", model or os.getenv("OPENAI_MODEL") or None, base_url, o_key, o_var)


def openai_client(cfg: LLMConfig):
    """Build an OpenAI SDK client for an OpenAI-compatible endpoint."""
    try:
        import openai
    except ImportError as exc:
        raise LLMUnavailable("the 'openai' package is not installed (pip install openai)") from exc
    if not cfg.api_key:
        raise LLMUnavailable(f"no API key for the OpenAI-compatible backend; set one of {', '.join(OPENAI_KEY_VARS)}")
    if not cfg.model:
        raise LLMUnavailable("no model set for the OpenAI-compatible backend; set LLM_MODEL (or pass --llm-model)")
    return openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0, max_retries=2)


def status_error_message(exc, cfg: LLMConfig) -> str:
    """One line with status, URL, model and the raw body: gateway errors often carry no JSON message."""
    body = (getattr(exc.response, "text", "") or "").strip().replace("\n", " ")[:300]
    req = getattr(exc, "request", None)
    where = f" from {req.method} {req.url}" if req is not None else ""
    msg = f"API error {exc.status_code}{where} (model={cfg.model!r}): {body or exc.message}"
    if exc.status_code == 404:
        msg += " - check the base URL (the SDK appends the API path itself) and that the model name exists"
    return msg
