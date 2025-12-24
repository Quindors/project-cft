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


def require_openai_api_key(openai_module, env_var: str = "OPENAI_API_KEY") -> None:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Set {env_var} in your environment (.env).")
    openai_module.api_key = key