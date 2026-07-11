# monitor/paths.py
"""Path resolution for bundled vs source-tree mode.

In source mode, files live in the project tree.
In frozen mode (PyInstaller onedir), bundled read-only files live under
sys._MEIPASS (the runtime unpack dir), and user-editable files (.env) live
next to sys.executable so the user can edit them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _project_root() -> Path:
    # monitor/paths.py -> monitor/ -> project root
    return Path(__file__).resolve().parent.parent


def _bundle_root() -> Path:
    """Root for bundled, read-only resources.

    Frozen: sys._MEIPASS (PyInstaller unpack dir).
    Source: project root.
    """
    return Path(getattr(sys, "_MEIPASS", _project_root()))


def exe_dir() -> Path:
    """Directory containing the running exe (frozen) or project root (source).

    User-editable files (.env, logs) belong here.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _project_root()


def resource_path(*parts: str) -> Path:
    """Resolve a bundled, read-only resource (e.g. the prompt files)."""
    return _bundle_root().joinpath(*parts)


def env_path() -> Path:
    """Location of the user-editable .env file."""
    return exe_dir() / ".env"


def data_dir() -> Path:
    """Writable per-user data directory (for the local SQLite DB, etc.).

    Honors CFT_DATA_DIR override; else %LOCALAPPDATA%\\cft on Windows (or
    $APPDATA), falling back to a data/ folder next to the exe.
    """
    override = os.environ.get("CFT_DATA_DIR")
    if override:
        d = Path(override)
    else:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        d = (Path(base) / "cft") if base else (exe_dir() / "data")
    d.mkdir(parents=True, exist_ok=True)
    return d
