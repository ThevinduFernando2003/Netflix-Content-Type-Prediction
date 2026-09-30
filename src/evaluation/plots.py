"""Charts for the evaluation report. One idea per figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve

from src.config import RANDOM_SEED

NAVY = "#1f4e79"
CLAY = "#c45911"
GREY = "#6b7280"


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion(raw: np.ndarray, normalised: np.ndarray, path: Path) -> None:
    """Side-by-side counts and row-normalised rates."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    labels = ["Movie", "TV Show"]
    for axis, matrix, title, fmt in (
        (axes[0], raw, "Confusion matrix (counts)", ".0f"),
        (axes[1], normalised, "Row-normalised rates", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks([0, 1], labels)
        axis.set_yticks([0, 1], labels)
        axis.set_xlabel("Predicted type")
        axis.set_ylabel("True type")
        axis.set_title(title)
        for row in range(2):
            for col in range(2):
                axis.text(
                    col, row, format(matrix[row, col], fmt), ha="center", va="center", color="black"
                )
        fig.colorbar(image, ax=axis, fraction=0.046)
    _finish(path)


def plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    dummy_score: np.ndarray,
    roc_path: Path,
    pr_path: Path,
) -> None:
    """ROC and precision-recall curves with the prior dummy on the same axes."""
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    RocCurveDisplay.from_predictions(y_true, y_score, name="honest model", ax=axis, color=NAVY)
    RocCurveDisplay.from_predictions(y_true, dummy_score, name="dummy prior", ax=axis, color=GREY)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curve (positive class: TV Show)")
    _finish(roc_path)

    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_score, name="honest model", ax=axis, color=NAVY
    )
    PrecisionRecallDisplay.from_predictions(
        y_true, dummy_score, name="dummy prior", ax=axis, color=GREY
    )
    axis.set_xlabel("Recall (TV Show)")
    axis.set_ylabel("Precision (TV Show)")
    axis.set_title("Precision-recall curve")
    _finish(pr_path)


def plot_calibration(
    y_true: np.ndarray,
    raw_score: np.ndarray,
    calibrated_score: np.ndarray,
    path: Path,
) -> None:
    """Reliability diagram before and after isotonic calibration."""
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    for score, label, color in (
        (raw_score, "before calibration", CLAY),
        (calibrated_score, "after isotonic calibration", NAVY),
    ):
        fraction, mean_predicted = calibration_curve(y_true, score, n_bins=8, strategy="quantile")
        axis.plot(mean_predicted, fraction, marker="o", color=color, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="perfect")
    axis.set_xlabel("Mean predicted P(TV Show)")
    axis.set_ylabel("Observed fraction of TV Shows")
    axis.set_title("Reliability diagram")
    axis.legend(frameon=False)
    _finish(path)


def plot_learning_curve(estimator: object, x: object, y: np.ndarray, path: Path) -> None:
    """Training and validation ROC-AUC against training size."""
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        x,
        y,
        cv=3,
        scoring="roc_auc",
        train_sizes=np.linspace(0.25, 1.0, 4),
        shuffle=True,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.plot(train_sizes, train_scores.mean(axis=1), color=CLAY, marker="o", label="train")
    axis.plot(train_sizes, valid_scores.mean(axis=1), color=NAVY, marker="o", label="validation")
    axis.set_xlabel("Training rows")
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Learning curve (3-fold)")
    axis.legend(frameon=False)
    _finish(path)


def plot_importance(names: list[str], drops: list[float], path: Path) -> None:
    """Horizontal bar chart of permutation AUC drop."""
    order = np.argsort(drops)
    fig_height = max(3.5, 0.28 * len(names))
    fig, axis = plt.subplots(figsize=(7.0, fig_height))
    axis.barh(np.array(names)[order], np.array(drops)[order], color=NAVY)
    axis.set_xlabel("ROC-AUC drop when the feature is permuted")
    axis.set_title("Permutation importance on the test set")
    _finish(path)


def plot_threshold(rows: list[dict[str, float]], path: Path) -> None:
    """F1 and TV-show precision/recall across thresholds."""
    threshold = [row["threshold"] for row in rows]
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    axis.plot(threshold, [row["precision_tv"] for row in rows], color=NAVY, label="TV precision")
    axis.plot(threshold, [row["recall_tv"] for row in rows], color=CLAY, label="TV recall")
    axis.plot(threshold, [row["f1_macro"] for row in rows], color=GREY, label="F1-macro")
    axis.set_xlabel("Threshold on P(TV Show)")
    axis.set_ylabel("Score")
    axis.set_title("Threshold sweep")
    axis.legend(frameon=False)
    _finish(path)
