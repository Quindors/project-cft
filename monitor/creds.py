# monitor/creds.py
from __future__ import annotations

import os
import subprocess
from dotenv import load_dotenv

from monitor.paths import env_path, is_frozen


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_env(*, override: bool = True) -> None:
    if is_frozen():
        load_dotenv(dotenv_path=str(env_path()), override=override)
    else:
        load_dotenv(override=override)


def _prompt_for_key(label: str) -> str:
    """Pop a Windows input dialog asking for an API key. Returns '' if cancelled."""
    ps = (
        "Add-Type -AssemblyName Microsoft.VisualBasic;"
        "[Microsoft.VisualBasic.Interaction]::InputBox("
        f"'project-cft needs your {label} API key to run. Paste it below - "
        "it will be saved to the .env next to the app so you are only asked once.',"
        "'project-cft setup','')"
    )
    try:
        # CREATE_NO_WINDOW: in the windowed exe, don't flash a console for powershell.
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=300, creationflags=no_window,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _persist_env_var(name: str, value: str) -> None:
    """Write or update NAME=value in the .env next to the app."""
    path = env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[creds] saved {name} to {path}")
    except Exception as e:
        print(f"[creds] could not save {name} to {path}: {e}")


def _require_key(var_name: str, label: str) -> str:
    """Return the API key from env, or prompt the user once and persist it."""
    key = os.getenv(var_name)
    if key:
        return key
    print(f"[creds] {var_name} not set - prompting user...")
    key = _prompt_for_key(label)
    if key:
        os.environ[var_name] = key
        _persist_env_var(var_name, key)
        return key
    raise RuntimeError(f"No {var_name} provided. Set it in your .env and restart.")


def configure_llm():
    """Build an OpenAI() client for the configured hosted provider.

    Returns (client, provider_name). If the required API key is missing, the
    user is prompted for it once and it's saved to the .env for next time.
    """
    from openai import OpenAI

    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        key = _require_key("OPENROUTER_API_KEY", "OpenRouter")
        headers = {"X-Title": os.getenv("OPENROUTER_TITLE", "project-cft")}
        ref = os.getenv("OPENROUTER_REFERER")
        if ref:
            headers["HTTP-Referer"] = ref
        client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, default_headers=headers)
    elif provider == "openai":
        key = _require_key("OPENAI_API_KEY", "OpenAI")
        client = OpenAI(api_key=key, max_retries=5)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r} (use 'openrouter' or 'openai')")

    return client, provider
