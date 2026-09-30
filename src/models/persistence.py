"""Save and load versioned model artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import repo_root, resolve_path
from src.logging_conf import get_logger

logger = get_logger(__name__)


def model_dir() -> Path:
    """Directory that holds joblib artifacts and metadata.json."""
    path = resolve_path(resolve_path_config())
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path_config() -> str:
    from src.config import get_config

    return str(get_config()["paths"]["model_dir"])


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed so large catalogues stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    """Current commit, or ``unknown`` when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    """Versions that affect a reproduced prediction."""
    import fastapi
    import numpy
    import pandas
    import sklearn

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "fastapi": fastapi.__version__,
    }


def save_artifacts(
    *,
    best_model: Any,
    calibrator: Any,
    label_encoder: dict[str, int],
    metadata: dict[str, Any],
    extra_models: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> Path:
    """Persist the served model, the calibrator, and metadata.json."""
    destination = directory or model_dir()
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, destination / "best_model.joblib")
    joblib.dump(calibrator, destination / "calibrator.joblib")
    joblib.dump(label_encoder, destination / "label_encoder.joblib")
    for name, model in (extra_models or {}).items():
        joblib.dump(model, destination / name)
    write_metadata(metadata, destination)
    logger.info("Saved artifacts to %s", destination)
    return destination


def write_metadata(metadata: dict[str, Any], directory: Path | None = None) -> Path:
    """Write metadata.json with stable key order."""
    destination = directory or model_dir()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_metadata(directory: Path | None = None) -> dict[str, Any]:
    folder = Path(directory) if directory is not None else model_dir()
    path = folder / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json is missing in {folder}. Run make train.")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def load_joblib(name: str, directory: Path | None = None) -> Any:
    folder = Path(directory) if directory is not None else model_dir()
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact {path}. Run make train.")
    return joblib.load(path)


def utc_now() -> str:
    """ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
