"""Engineered feature lists for the honest, leaky, and ablated models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.schema import synthesized_columns


@dataclass(frozen=True)
class FeatureSpec:
    """Columns the ColumnTransformer is allowed to read."""

    text_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    extra_categorical: tuple[str, ...]
    strip_tv_vocabulary: bool
    name: str

    @property
    def all_columns(self) -> tuple[str, ...]:
        return (
            *self.text_columns,
            *self.categorical_columns,
            *self.numeric_columns,
            *self.extra_categorical,
        )


def _base_numeric(frame: pd.DataFrame) -> list[str]:
    columns = [
        "release_year",
        "year_added",
        "month_added",
        "years_to_platform",
        "n_genres",
        "n_countries",
        "title_word_count",
    ]
    synthesized = set(synthesized_columns(frame))
    if "description" not in synthesized:
        columns.extend(["description_char_len", "description_word_count"])
    if "cast" not in synthesized:
        columns.append("n_cast")
    flags = get_config()["features"]
    if bool(flags.get("use_has_director", False)):
        columns.append("has_director")
    return columns


def _base_categorical() -> list[str]:
    columns = ["primary_country"]
    if bool(get_config()["features"].get("use_rating", False)):
        columns.append("rating")
    return columns


def _base_text(frame: pd.DataFrame) -> list[str]:
    columns = ["title"]
    if "description" not in set(synthesized_columns(frame)):
        columns.append("description")
    return columns


def honest_spec(frame: pd.DataFrame) -> FeatureSpec:
    """Non-leaky features. Duration and genre names are not included."""
    return FeatureSpec(
        text_columns=tuple(_base_text(frame)),
        categorical_columns=tuple(_base_categorical()),
        numeric_columns=tuple(_base_numeric(frame)),
        extra_categorical=(),
        strip_tv_vocabulary=False,
        name="honest",
    )


def no_tv_vocabulary_spec(frame: pd.DataFrame) -> FeatureSpec:
    """Honest features with format-naming words stripped from text fields."""
    spec = honest_spec(frame)
    return FeatureSpec(
        text_columns=spec.text_columns,
        categorical_columns=spec.categorical_columns,
        numeric_columns=spec.numeric_columns,
        extra_categorical=(),
        strip_tv_vocabulary=True,
        name="no_tv_vocabulary",
    )


def leaky_spec(frame: pd.DataFrame) -> FeatureSpec:
    """Honest features plus ``duration_unit``, the perfect separator."""
    spec = honest_spec(frame)
    return FeatureSpec(
        text_columns=spec.text_columns,
        categorical_columns=spec.categorical_columns,
        numeric_columns=spec.numeric_columns,
        extra_categorical=("duration_unit",),
        strip_tv_vocabulary=False,
        name="leaky",
    )


def _missing_category(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip() == "" or str(value).strip().lower() in {
        "not given",
        "unknown",
        "nan",
        "none",
    }


def prepare_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return the model matrix with missing categoricals as NaN and text as strings."""
    out = frame.loc[:, list(spec.all_columns)].copy()
    for column in spec.text_columns:
        out[column] = out[column].map(lambda value: "" if _missing_category(value) else str(value))
    for column in (*spec.categorical_columns, *spec.extra_categorical):
        out[column] = out[column].map(
            lambda value: np.nan if _missing_category(value) else str(value)
        )
    return out


def model_frame(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return the columns a pipeline reads, including the label when present."""
    columns = list(spec.all_columns)
    if "type" in frame.columns:
        columns.append("type")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Feature frame is missing columns: {missing}")
    return frame.loc[:, columns].copy()
