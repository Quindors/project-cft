# monitor/creds.py
from __future__ import annotations

import os
import re
import subprocess
from dotenv import load_dotenv

from monitor.paths import env_path, is_frozen


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_env(*, override: bool = True) -> None:
    # utf-8-sig: tolerate a UTF-8 BOM (e.g. from PowerShell's Set-Content),
    # which would otherwise silently corrupt the first line's variable name.
    if is_frozen():
        load_dotenv(dotenv_path=str(env_path()), override=override, encoding="utf-8-sig")
    else:
        load_dotenv(override=override, encoding="utf-8-sig")


def _prompt_for_key(label: str) -> str:
    """Pop a Windows input dialog asking for an API key. Returns '' if cancelled."""
    ps = (
        "Add-Type -AssemblyName Microsoft.VisualBasic;"
        "[Microsoft.VisualBasic.Interaction]::InputBox("
        f"'cft needs a {label} API key before it can classify what you are "
        f"working on. Paste it below and you will not be asked again - it is "
        f"saved to the .env file next to the app, on this PC only. "
        f"Without a key, cft still records which windows you use, but cannot "
        f"sort them into categories.',"
        "'Set up cft','')"
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


# Characters a value can carry bare on a KEY=value line. Anything else —
# spaces, `#` (starts a comment mid-value), quotes, backslashes — has to be
# quoted or python-dotenv reads back something other than what was saved.
# Tokens, slugs, URLs and flags are all bare; an SMTP password is the case
# that is not.
_BARE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+=,%~\-]*$")


def _env_literal(value: str) -> str:
    """`value` as a .env literal that load_dotenv reads back verbatim."""
    if _BARE_VALUE.match(value):
        return value
    if "$" in value and "'" not in value:
        # `${NAME}` is interpolated inside double quotes; single quotes are
        # the only way to keep a dollar sign literal.
        return "'" + value + "'"
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
                    .replace("\n", "\\n").replace("\r", "\\r"))
    return '"' + escaped + '"'


def _persist_env_var(name: str, value: str) -> None:
    """Write or update NAME=value in the .env next to the app."""
    path = env_path()
    literal = _env_literal(value)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={literal}"
            break
    else:
        lines.append(f"{name}={literal}")
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[creds] saved {name} to {path}")
    except Exception as e:
        print(f"[creds] Couldn't save {name} to {path}. It will work for "
              f"this session but you'll be asked again next time \u2014 check "
              f"the app's folder isn't read-only. ({e})")


def _require_key(var_name: str, label: str) -> str:
    """Return the API key from env, or prompt the user once and persist it."""
    key = os.getenv(var_name)
    if key:
        return key
    print(f"[creds] No {var_name} on file yet; asking for one.")
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
