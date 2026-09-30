"""Engineered features and the banned-column guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.clean import clean_catalogue
from src.data.load import load_raw_csv
from src.features.engineer import honest_spec, leaky_spec, no_tv_vocabulary_spec, prepare_features
from src.features.inference import build_inference_frame
from src.models.train import assert_honest_inputs
from tests.conftest import FIXTURE


def test_numeric_features_on_the_fixture(catalogue: pd.DataFrame) -> None:
    row = catalogue.loc[catalogue["show_id"] == "m8"].iloc[0]
    assert row["n_genres"] == 5
    assert row["n_cast"] == 3
    assert row["title_word_count"] == 2
    assert row["years_to_platform"] == row["year_added"] - row["release_year"]
    assert row["has_director"] == 1


def test_missing_indicator_inputs_are_nan(catalogue: pd.DataFrame) -> None:
    spec = honest_spec(catalogue)
    frame = prepare_features(catalogue, spec)
    no_country = catalogue.index[catalogue["show_id"] == "m4"][0]
    prepared_index = frame.index[catalogue["show_id"] == "m4"][0]
    del no_country
    assert pd.isna(frame.loc[prepared_index, "primary_country"])
    assert "duration" not in frame.columns
    assert "has_director" not in frame.columns


def test_synthesized_text_columns_are_dropped() -> None:
    raw = load_raw_csv(FIXTURE).drop(columns=["description", "cast"])
    cleaned = clean_catalogue(raw)
    spec = honest_spec(cleaned)
    assert "description" not in spec.text_columns
    assert "description_word_count" not in spec.numeric_columns
    assert "n_cast" not in spec.numeric_columns


def test_honest_spec_rejects_duration(catalogue: pd.DataFrame) -> None:
    assert_honest_inputs(honest_spec(catalogue))
    assert_honest_inputs(no_tv_vocabulary_spec(catalogue))
    try:
        assert_honest_inputs(leaky_spec(catalogue))
        raise AssertionError("leaky spec should not pass the honest guard")
    except RuntimeError as exc:
        assert "duration_unit" in str(exc)


def test_inference_frame_imputes_missing_dates() -> None:
    metadata = {
        "spec": {
            "text": ["title"],
            "categorical": ["primary_country", "rating"],
            "numeric": ["release_year", "year_added", "n_genres", "title_word_count"],
            "extra_categorical": [],
        }
    }
    frame = build_inference_frame(
        {
            "title": "Harbor Heist",
            "description": "unused but long enough",
            "release_year": 2019,
            "n_genres": 2,
        },
        metadata,
    )
    assert frame.loc[0, "title"] == "Harbor Heist"
    assert frame.loc[0, "title_word_count"] == 2
    assert np.isnan(frame.loc[0, "year_added"])
    assert frame.loc[0, "n_genres"] == 2
