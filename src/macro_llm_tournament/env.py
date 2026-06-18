from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_PATHS = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[2] / ".env",
]


def load_secret_env() -> list[str]:
    loaded: list[str] = []
    for path in DEFAULT_ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=False)
            loaded.append(str(path))
    return loaded


def present_secret_names() -> list[str]:
    return [key for key in ["FRED_API_KEY"] if os.getenv(key)]
