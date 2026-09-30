"""Stratified splitting. The test set is cut once, before any fitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, get_config


@dataclass(frozen=True)
class SplitData:
    """Hold-out split plus the labels."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def encode_label(y: pd.Series) -> pd.Series:
    """Map Movie -> 0 and TV Show -> 1. The positive class is the minority class."""
    mapping = {"Movie": 0, "TV Show": 1}
    unknown = sorted(set(y.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    return y.map(mapping).astype(int)


def stratified_split(frame: pd.DataFrame, test_size: float | None = None) -> SplitData:
    """Split ``frame`` into a stratified train and test set.

    ``test_size`` defaults to the config value (0.15). The seed is ``RANDOM_SEED``.
    """
    if test_size is None:
        test_size = float(get_config()["split"]["test_size"])
    features = frame.drop(columns=["type"])
    labels = frame["type"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    return SplitData(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
