"""Probability calibration and the operating threshold.

The threshold is chosen on out-of-fold probabilities, never on the test set.
Isotonic regression is fit inside each training fold of those out-of-fold scores.
The served calibrator is a separate ``CalibratedClassifierCV`` fit on all training rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import RANDOM_SEED, get_config
from src.logging_conf import get_logger

logger = get_logger(__name__)


def fold_splitter(y: pd.Series | np.ndarray) -> StratifiedKFold:
    """The same 5-fold splitter used for model selection."""
    return StratifiedKFold(
        n_splits=int(get_config()["cv"]["n_splits"]),
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def out_of_fold_probabilities(
    estimator: Any, x: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and fold-wise isotonic-calibrated P(TV Show)."""
    y_array = np.asarray(y)
    folds = fold_splitter(y_array)
    raw = cross_val_predict(
        clone(estimator),
        x,
        y_array,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    calibrated = np.zeros_like(raw)
    for train_idx, valid_idx in folds.split(x, y_array):
        if len(np.unique(y_array[train_idx])) < 2 or len(np.unique(raw[train_idx])) < 3:
            calibrated[valid_idx] = raw[valid_idx]
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[train_idx], y_array[train_idx])
        calibrated[valid_idx] = iso.predict(raw[valid_idx])
    return raw, np.clip(calibrated, 0.0, 1.0)


def fit_calibrator(estimator: Any, x: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Isotonic calibration fitted only on the training rows.

    ``ensemble=False`` still uses 5-fold cross-validated scores to fit the
    isotonic map, then one model on all training rows for prediction. The
    default ensemble would serve five full pipelines and miss the latency budget.
    """
    calibrator = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="isotonic",
        cv=fold_splitter(y),
        ensemble=False,
    )
    calibrator.fit(x, y)
    return calibrator


def tune_threshold(y: pd.Series | np.ndarray, probability_tv: np.ndarray) -> dict[str, Any]:
    """Pick the F1-macro threshold and the highest-recall threshold at 95% TV precision."""
    y_array = np.asarray(y)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows: list[dict[str, float]] = []
    best_f1 = -1.0
    best_threshold = 0.5
    precision_target = float(get_config()["threshold"]["high_precision_target"])
    high_precision: list[tuple[float, float]] = []
    for threshold in grid:
        predicted = (probability_tv >= threshold).astype(int)
        f1_macro = float(f1_score(y_array, predicted, average="macro", zero_division=0))
        precision_tv = float(precision_score(y_array, predicted, pos_label=1, zero_division=0))
        recall_tv = float(recall_score(y_array, predicted, pos_label=1, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "f1_macro": f1_macro,
                "precision_tv": precision_tv,
                "recall_tv": recall_tv,
            }
        )
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = float(threshold)
        if precision_tv >= precision_target and predicted.sum() > 0:
            high_precision.append((recall_tv, float(threshold)))
    high_precision_threshold = max(high_precision)[1] if high_precision else None
    logger.info(
        "Tuned threshold %.2f (F1-macro %.4f). High-precision threshold: %s",
        best_threshold,
        best_f1,
        high_precision_threshold,
    )
    return {
        "threshold": best_threshold,
        "f1_macro": best_f1,
        "high_precision_threshold": high_precision_threshold,
        "sweep": rows,
    }
