# monitor/creds.py
from __future__ import annotations

import os, base64, tempfile
from dotenv import load_dotenv


def load_env(*, override: bool = True) -> None:
    load_dotenv(override=override)


def build_service_account_json_from_b64(env_var: str = "GCP_SERVICE_ACCOUNT_B64") -> str:
    b64 = os.getenv(env_var)
    if not b64:
        raise RuntimeError(f"{env_var} not set in environment (.env).")

    data = base64.b64decode(b64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def configure_llm(openai_module) -> str:
    """Point the openai SDK at OpenAI or OpenRouter based on LLM_PROVIDER.
    Returns the provider name so the caller can log it.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your .env")
        openai_module.api_key = key
        openai_module.base_url = OPENROUTER_BASE_URL
        # Optional — shows up on the OpenRouter activity dashboard
        headers = {"X-Title": os.getenv("OPENROUTER_TITLE", "project-cft")}
        ref = os.getenv("OPENROUTER_REFERER")
        if ref:
            headers["HTTP-Referer"] = ref
        openai_module.default_headers = headers
    elif provider == "openai":
        require_openai_api_key(openai_module)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r} (expected 'openrouter' or 'openai')")
    return provider

def require_openai_api_key(openai_module, env_var: str = "OPENAI_API_KEY") -> None:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Set {env_var} in your environment (.env).")
    openai_module.api_key = key