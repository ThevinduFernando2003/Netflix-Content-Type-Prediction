"""Local explanations and the error-analysis hypotheses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.logging_conf import get_logger

logger = get_logger(__name__)


def _dense_row(matrix: Any, index: int = 0) -> np.ndarray:
    row = matrix[index]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def linear_contributions(
    pipeline: Any, frame: pd.DataFrame, limit: int = 8
) -> list[dict[str, Any]]:
    """Token contributions from a linear classifier: coefficient times feature value.

    The positive direction is TV Show, which is ``classes_[1]`` for a 0/1 label.
    """
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return []
    transformed = pipeline.named_steps["preprocess"].transform(frame)
    names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    coefficients = np.asarray(classifier.coef_).ravel()
    row = _dense_row(transformed)
    if row.shape[0] != coefficients.shape[0]:
        logger.warning("Coefficient vector length does not match the transformed row.")
        return []
    contribution = row * coefficients
    order = np.argsort(np.abs(contribution))[::-1][:limit]
    results = []
    for index in order:
        value = float(contribution[index])
        if value == 0.0:
            continue
        results.append(
            {
                "feature": names[index],
                "contribution": value,
                "direction": "tv_show" if value > 0 else "movie",
            }
        )
    return results


def permutation_importance_top(
    pipeline: Any,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    *,
    limit: int = 25,
    n_repeats: int = 5,
) -> list[dict[str, float | str]]:
    """Permute the transformed features with the largest absolute coefficients.

    Importance is the drop in test ROC-AUC. The preprocessor stays frozen.
    """
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_") or not hasattr(classifier, "predict_proba"):
        return []
    transformed = pipeline.named_steps["preprocess"].transform(frame)
    names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    coefficients = np.abs(np.asarray(classifier.coef_).ravel())
    chosen = np.argsort(coefficients)[::-1][:limit]
    baseline = float(roc_auc_score(y_true, classifier.predict_proba(transformed)[:, 1]))
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | str]] = []
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    for index in chosen:
        drops: list[float] = []
        original = dense[:, index].copy()
        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            dense[:, index] = shuffled
            score = float(roc_auc_score(y_true, classifier.predict_proba(dense)[:, 1]))
            drops.append(baseline - score)
        dense[:, index] = original
        rows.append({"feature": names[int(index)], "auc_drop": float(np.mean(drops))})
    rows.sort(key=lambda item: float(item["auc_drop"]), reverse=True)
    return rows


def top_coefficients(pipeline: Any, per_class: int = 20) -> dict[str, list[dict[str, float | str]]]:
    """Largest positive and negative coefficients, independent of a single row."""
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return {"tv_show": [], "movie": []}
    names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    coefficients = np.asarray(classifier.coef_).ravel()
    order = np.argsort(coefficients)
    movie: list[dict[str, float | str]] = [
        {"feature": names[int(index)], "coefficient": float(coefficients[int(index)])}
        for index in order[:per_class]
    ]
    tv_show: list[dict[str, float | str]] = [
        {"feature": names[int(index)], "coefficient": float(coefficients[int(index)])}
        for index in order[::-1][:per_class]
    ]
    return {"tv_show": tv_show, "movie": movie}


def column_permutation(
    model: Any,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    *,
    n_repeats: int = 5,
) -> list[dict[str, float | str]]:
    """Permute each raw input column and record the drop in ROC-AUC."""
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    tv_index = classes.index(1)
    baseline = float(roc_auc_score(y_true, probabilities[:, tv_index]))
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | str]] = []
    for column in frame.columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = float(roc_auc_score(y_true, model.predict_proba(shuffled)[:, tv_index]))
            drops.append(baseline - score)
        rows.append({"feature": column, "auc_drop": float(np.mean(drops))})
    rows.sort(key=lambda item: float(item["auc_drop"]), reverse=True)
    return rows


def hypothesis_for_row(row: pd.Series, predicted: str) -> str:
    """One-line reason a confident mistake is plausible from the columns we kept."""
    title = str(row.get("title", ""))
    genres = row.get("genres", [])
    genre_text = " ".join(genres) if isinstance(genres, list) else str(genres)
    if bool(row.get("genre_has_banned_tv_token", 0)) and predicted == "Movie":
        return "The listing uses a TV-format genre the honest model is not allowed to see, and the title does not say so."
    if float(row.get("has_director", 1) or 0) == 0 and predicted == "Movie":
        return "No director is credited, which in this catalogue usually means a series, but that flag is banned."
    if len(title.split()) <= 2:
        return "The title is very short, so the text model has almost nothing to read."
    rating = str(row.get("rating", ""))
    if rating in {"TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G"} and predicted == "Movie":
        return f"The rating {rating} leans toward TV, but the rest of the features outweighed it."
    if "documentary" in genre_text.casefold() or "docuseries" in genre_text.casefold():
        return "Documentaries and docuseries share a subject register; only the banned genre name separates them."
    if "stand-up" in genre_text.casefold() or "stand-up" in title.casefold():
        return "Stand-up specials are one-off films that read like episodes of a comedy series."
    years = row.get("years_to_platform")
    if isinstance(years, float) and years == years and years >= 10 and predicted == "TV Show":
        return "A long gap between release and the platform date is more common for library films."
    if not str(row.get("description", "")).strip():
        return "No synopsis was available, so the model is guessing from the title and a few catalogue fields."
    return "The title wording and catalogue fields point the other way."
