"""Charts and local explanations on a tiny fitted model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED
from src.data.split import encode_label, stratified_split
from src.evaluation.explain import (
    column_permutation,
    hypothesis_for_row,
    linear_contributions,
    permutation_importance_top,
    top_coefficients,
)
from src.evaluation.metrics import score_predictions, segment_table, text_report, threshold_sweep
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_roc_pr,
    plot_threshold,
)
from src.features.build_pipeline import TVVocabularyStripper, build_model_pipeline
from src.features.engineer import honest_spec, prepare_features


def test_scores_and_plots(tmp_path: Path, catalogue) -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.2, 0.6, 0.7, 0.9])
    metrics = score_predictions(y_true, y_pred, y_score)
    assert 0 <= metrics["roc_auc"] <= 1
    assert "TV Show" in text_report(y_true, y_pred)
    assert threshold_sweep(y_true, y_score)
    frame = __import__("pandas").DataFrame({"bucket": ["a", "a", "b", "b"]})
    segments = segment_table(frame, y_true, y_pred, "bucket", overall=0.9)
    assert segments
    plot_confusion(
        np.array([[2, 1], [0, 1]]), np.array([[0.67, 0.33], [0.0, 1.0]]), tmp_path / "cm.png"
    )
    plot_roc_pr(
        y_true, y_score, np.array([0.5, 0.5, 0.5, 0.5]), tmp_path / "roc.png", tmp_path / "pr.png"
    )
    plot_calibration(y_true, y_score, y_score, tmp_path / "cal.png")
    plot_threshold(threshold_sweep(y_true, y_score), tmp_path / "thr.png")
    plot_importance(["title"], [0.1], tmp_path / "imp.png")
    assert (tmp_path / "imp.png").exists()


def test_stripper_and_contributions(catalogue) -> None:
    stripper = TVVocabularyStripper(stoplist=["series", "season"], enabled=True)
    stripper.fit(["placeholder"])
    cleaned = stripper.transform(["this series returns next season"])
    assert "series" not in cleaned[0].casefold()
    assert "season" not in cleaned[0].casefold()

    spec = honest_spec(catalogue)
    split = stratified_split(catalogue, test_size=0.25)
    pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=300, solver="liblinear", random_state=RANDOM_SEED),
    )
    x_train = prepare_features(split.x_train, spec)
    pipeline.fit(x_train, encode_label(split.y_train))
    contributions = linear_contributions(pipeline, x_train.iloc[[0]])
    assert contributions
    assert contributions[0]["direction"] in {"movie", "tv_show"}
    y_train = encode_label(split.y_train).to_numpy()
    assert permutation_importance_top(pipeline, x_train, y_train, limit=3, n_repeats=1)
    assert top_coefficients(pipeline, per_class=3)["tv_show"]
    assert column_permutation(pipeline, x_train, y_train, n_repeats=1)


def test_hypothesis_mentions_the_banned_genre() -> None:
    import pandas as pd

    row = pd.Series(
        {
            "title": "Night",
            "genres": ["TV Dramas"],
            "genre_has_banned_tv_token": 1,
            "has_director": 0,
            "rating": "TV-MA",
            "description": "",
            "years_to_platform": 1,
        }
    )
    text = hypothesis_for_row(row, "Movie")
    assert "TV-format genre" in text
