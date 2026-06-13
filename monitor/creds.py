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


def configure_llm(_openai_module=None):
    """Build and return an OpenAI() client pointing at the configured provider.

    The caller should use the returned client object directly (client.chat.
    completions.create(...)) rather than the module-level openai.* calls,
    because module-level base_url/api_key are flaky across SDK versions.

    Returns (client, provider_name).
    """
    from openai import OpenAI

    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your .env")
        headers = {"X-Title": os.getenv("OPENROUTER_TITLE", "project-cft")}
        ref = os.getenv("OPENROUTER_REFERER")
        if ref:
            headers["HTTP-Referer"] = ref
        client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, default_headers=headers)
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY in your .env")
        client = OpenAI(api_key=key)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r}")

    return client, provider

def require_openai_api_key(openai_module, env_var: str = "OPENAI_API_KEY") -> None:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Set {env_var} in your environment (.env).")
    openai_module.api_key = key