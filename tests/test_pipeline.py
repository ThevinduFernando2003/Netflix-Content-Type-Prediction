"""Pipeline behaviour: probabilities, unknown categories, fold isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED
from src.data.split import encode_label, stratified_split
from src.features.build_pipeline import build_model_pipeline
from src.features.engineer import honest_spec, prepare_features


def _fit(catalogue: pd.DataFrame):
    spec = honest_spec(catalogue)
    split = stratified_split(catalogue, test_size=0.25)
    pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=400, solver="liblinear", random_state=RANDOM_SEED),
    )
    x_train = prepare_features(split.x_train, spec)
    y_train = encode_label(split.y_train)
    pipeline.fit(x_train, y_train)
    return pipeline, spec, split


def test_probabilities_sum_to_one_and_single_row_works(catalogue) -> None:
    pipeline, spec, split = _fit(catalogue)
    frame = prepare_features(split.x_test, spec)
    probabilities = pipeline.predict_proba(frame)
    assert probabilities.shape[1] == 2
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    single = pipeline.predict_proba(frame.iloc[[0]])
    assert single.shape == (1, 2)


def test_unknown_category_does_not_crash(catalogue) -> None:
    pipeline, spec, split = _fit(catalogue)
    frame = prepare_features(split.x_test, spec)
    frame.loc[frame.index[0], "primary_country"] = "Narnia"
    frame.loc[frame.index[0], "rating"] = "NOT-A-RATING"
    probabilities = pipeline.predict_proba(frame.iloc[[0]])
    assert probabilities.shape == (1, 2)


def test_sentinel_token_stays_out_of_the_training_fold(catalogue) -> None:
    spec = honest_spec(catalogue)
    split = stratified_split(catalogue, test_size=0.25)
    x_train = prepare_features(split.x_train, spec)
    x_test = prepare_features(split.x_test, spec)
    x_test = x_test.copy()
    x_test["description"] = "zyxsentinel zyxsentinel story"
    y_train = encode_label(split.y_train)
    pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_SEED),
    )
    pipeline.fit(x_train, y_train)
    vocabulary = set(
        pipeline.named_steps["preprocess"]
        .named_transformers_["description"]
        .named_steps["tfidf"]
        .vocabulary_
    )
    assert "zyxsentinel" not in vocabulary

    full = pd.concat([x_train, x_test], ignore_index=True)
    full_y = pd.concat([y_train, encode_label(split.y_test)], ignore_index=True)
    # The token appears in every test row, so a full-data fit clears min_df.
    full_pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_SEED),
    )
    full_pipeline.fit(full, full_y)
    full_vocabulary = set(
        full_pipeline.named_steps["preprocess"]
        .named_transformers_["description"]
        .named_steps["tfidf"]
        .vocabulary_
    )
    assert "zyxsentinel" in full_vocabulary
