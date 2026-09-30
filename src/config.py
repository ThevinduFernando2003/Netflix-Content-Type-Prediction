"""Project configuration loaded from ``config.yaml``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RANDOM_SEED: int = 42

MISSING_TOKENS: frozenset[str] = frozenset(
    {"", "not given", "unknown", "n/a", "na", "none", "null", "nan", "-"}
)


def repo_root() -> Path:
    """Return the repository root (the directory that contains ``config.yaml``)."""
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    return here.parents[1]


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load ``config.yaml`` once and cache it."""
    path = repo_root() / "config.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a mapping, got {type(data).__name__}")
    return data


def get_config() -> dict[str, Any]:
    """Return the cached config, reloading when tests clear the cache."""
    return load_config()
