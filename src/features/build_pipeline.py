"""Scikit-learn pipelines. Every learned transform lives inside the Pipeline."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_config
from src.features.engineer import FeatureSpec


def column_to_text(values: Any) -> list[str]:
    """Flatten a column-transformer text block into a list of strings."""
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
        return series.fillna("").astype(str).tolist()
    if isinstance(values, pd.Series):
        return values.fillna("").astype(str).tolist()
    array = np.asarray(values, dtype=object)
    if array.ndim == 2:
        array = array[:, 0]
    output: list[str] = []
    for value in array.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            output.append("")
        else:
            output.append(str(value))
    return output


class TVVocabularyStripper(BaseEstimator, TransformerMixin):
    """Drop a fixed format-word stoplist. Nothing is learned in ``fit``."""

    def __init__(self, stoplist: list[str] | None = None, enabled: bool = False) -> None:
        self.stoplist = stoplist
        self.enabled = enabled

    def fit(self, x: Any, y: Any = None) -> "TVVocabularyStripper":
        del y
        words = (
            self.stoplist
            if self.stoplist is not None
            else list(get_config()["leakage"]["tv_vocabulary"])
        )
        self.words_: list[str] = [word.casefold() for word in words]
        pattern = r"\b(" + "|".join(re.escape(word) for word in self.words_) + r")\b"
        self.pattern_: re.Pattern[str] = re.compile(pattern, flags=re.IGNORECASE)
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(["text"], dtype=object)
        return np.asarray(input_features, dtype=object)

    def transform(self, x: Any) -> list[str]:
        texts = column_to_text(x)
        if not self.enabled:
            return texts
        return [self.pattern_.sub(" ", text) for text in texts]


def _text_pipeline(max_features: int, strip: bool, svd_components: int | None) -> Pipeline:
    cfg = get_config()["features"]
    steps: list[tuple[str, BaseEstimator]] = [
        (
            "strip",
            TVVocabularyStripper(
                stoplist=list(get_config()["leakage"]["tv_vocabulary"]),
                enabled=strip,
            ),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, int(cfg["ngram_max"])),
                min_df=int(cfg["min_df"]),
                max_features=max_features,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                dtype=np.float32,
            ),
        ),
    ]
    if svd_components is not None:
        steps.append(
            (
                "svd",
                TruncatedSVD(
                    n_components=int(svd_components), random_state=int(get_config()["seed"])
                ),
            )
        )
    return Pipeline(steps=steps)


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def _one_hot() -> OneHotEncoder:
    cfg = get_config()["features"]
    return OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=int(cfg["one_hot_min_frequency"]),
        dtype=np.float32,
        sparse_output=True,
    )


class Densifier(BaseEstimator, TransformerMixin):
    """Turn a sparse matrix into a dense array for tree ensembles."""

    def fit(self, x: Any, y: Any = None) -> "Densifier":
        del x, y
        return self

    def transform(self, x: Any) -> np.ndarray:
        if hasattr(x, "toarray"):
            return np.asarray(x.toarray(), dtype=np.float32)
        return np.asarray(x, dtype=np.float32)


def build_preprocessor(
    spec: FeatureSpec,
    *,
    scale_numeric: bool = True,
    svd_components: int | None = None,
) -> ColumnTransformer:
    """Build a fold-safe column transformer for ``spec``.

    ``scale_numeric=False`` keeps imputed numerics unscaled. Naive Bayes does not
    use this preprocessor: it is restricted to non-negative text and one-hot
    features because ``years_to_platform`` can be negative.

    ``svd_components`` reduces each text block inside the fold for tree models.
    """
    cfg = get_config()["features"]
    transformers: list[tuple[str, Any, Any]] = []
    for column in spec.text_columns:
        max_features = (
            int(cfg["description_max_features"])
            if column == "description"
            else int(cfg["title_max_features"])
        )
        transformers.append(
            (column, _text_pipeline(max_features, spec.strip_tv_vocabulary, svd_components), column)
        )

    categorical = [*spec.categorical_columns, *spec.extra_categorical]
    if categorical:
        transformers.append(("categorical", _one_hot(), categorical))

    if spec.numeric_columns and scale_numeric:
        transformers.append(("numeric", _numeric_pipeline(), list(spec.numeric_columns)))
    elif spec.numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                list(spec.numeric_columns),
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def build_model_pipeline(
    spec: FeatureSpec,
    estimator: BaseEstimator,
    *,
    scale_numeric: bool = True,
    svd_components: int | None = None,
    densify: bool = False,
) -> Pipeline:
    """Attach ``estimator`` behind a preprocessor fitted only inside ``fit``."""
    steps: list[tuple[str, BaseEstimator]] = [
        (
            "preprocess",
            build_preprocessor(spec, scale_numeric=scale_numeric, svd_components=svd_components),
        ),
    ]
    if densify:
        steps.append(("dense", Densifier()))
    steps.append(("clf", estimator))
    return Pipeline(steps=steps)
