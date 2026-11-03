"""Load config/settings.yaml and resolve project paths."""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Read settings.yaml once and return it as a dict."""
    with open(SETTINGS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in settings, e.g. path("processed")."""
    return PROJECT_ROOT / load_settings()["paths"][key]
