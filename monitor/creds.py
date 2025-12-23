# focusmon/creds.py
import os
import base64
import tempfile
import atexit
from dotenv import load_dotenv

def load_env(*, override: bool = True) -> None:
    load_dotenv(override=override)

def build_service_account_json_from_b64(env_var: str = "GCP_SERVICE_ACCOUNT_B64") -> str:
    b64 = os.getenv(env_var)
    if not b64:
        raise RuntimeError(f"{env_var} not set in .env")

    data = base64.b64decode(b64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    path = tmp.name

    def _cleanup(p=path):
        try:
            os.remove(p)
        except OSError:
            pass

    atexit.register(_cleanup)
    return path

def require_openai_api_key(openai_module) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    openai_module.api_key = key
    return key
