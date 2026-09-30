"""Load the served model once and score a single frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logging_conf import get_logger
from src.models.persistence import load_joblib, load_metadata

logger = get_logger(__name__)


@dataclass
class Runtime:
    """Artifacts the API keeps in memory."""

    model: Any
    calibrator: Any
    label_encoder: dict[str, int]
    metadata: dict[str, Any]
    directory: Path

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))


def load_runtime(directory: Path | None = None) -> Runtime | None:
    """Load artifacts. Return None when training has not been run yet."""
    folder = directory
    try:
        metadata = load_metadata(folder)
        model = load_joblib("best_model.joblib", folder)
        calibrator = load_joblib("calibrator.joblib", folder)
        encoder = load_joblib("label_encoder.joblib", folder)
    except FileNotFoundError as exc:
        logger.warning("Model artifacts are not available: %s", exc)
        return None
    resolved = Path(metadata.get("model_dir", folder or "."))
    logger.info(
        "Loaded %s trained_at=%s git_sha=%s threshold=%s",
        metadata.get("model_name"),
        metadata.get("trained_at"),
        metadata.get("git_sha"),
        metadata.get("threshold"),
    )
    return Runtime(
        model=model,
        calibrator=calibrator,
        label_encoder={str(key): int(value) for key, value in encoder.items()},
        metadata=metadata,
        directory=resolved,
    )


def probability_tv(runtime: Runtime, frame: pd.DataFrame) -> np.ndarray:
    """Calibrated P(TV Show) for each row of ``frame``."""
    probabilities = runtime.calibrator.predict_proba(frame)
    classes = list(runtime.calibrator.classes_)
    tv_index = classes.index(1)
    return np.clip(np.asarray(probabilities)[:, tv_index], 0.0, 1.0)


def predict_labels(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the tuned threshold. 1 is TV Show."""
    return (probability >= threshold).astype(int)
