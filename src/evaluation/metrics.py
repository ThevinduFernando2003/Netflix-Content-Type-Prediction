"""Metric helpers used by the evaluation report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Threshold-free scores plus the scores of a concrete prediction."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_tv": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_movie": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_tv": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_tv": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_movie": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_movie": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, list[list[int]]]:
    """Raw and row-normalised confusion matrices ordered as Movie, TV Show."""
    raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = raw.sum(axis=1, keepdims=True)
    normal = np.divide(raw, np.maximum(row_sums, 1), dtype=float)
    return {"raw": raw.astype(int).tolist(), "normalised": normal.round(4).tolist()}


def text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Sklearn classification report with explicit class names."""
    return str(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Movie", "TV Show"],
            digits=4,
            zero_division=0,
        )
    )


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float]]:
    """Precision, recall, and F1 from 0.30 to 0.70."""
    rows: list[dict[str, float]] = []
    for threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
        predicted = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision_tv": float(
                    precision_score(y_true, predicted, pos_label=1, zero_division=0)
                ),
                "recall_tv": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
                "f1_tv": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
                "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_true, predicted)),
            }
        )
    return rows


def segment_table(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column: str,
    *,
    overall: float,
) -> list[dict[str, Any]]:
    """Accuracy by a segment column. Flag segments materially below the overall accuracy."""
    rows: list[dict[str, Any]] = []
    working = frame.copy()
    working["_y"] = y_true
    working["_pred"] = y_pred
    for key, group in working.groupby(column, dropna=False):
        accuracy = float((group["_y"] == group["_pred"]).mean()) if len(group) else float("nan")
        rows.append(
            {
                "segment": str(key),
                "support": int(len(group)),
                "accuracy": accuracy,
                "flagged": bool(accuracy < overall - 0.08 and len(group) >= 20),
            }
        )
    rows.sort(key=lambda item: item["support"], reverse=True)
    return rows
