"""Dataset contract.

The shipped Excel catalogue does not match the public Kaggle Netflix file.
Required columns are the ones that file actually contains. ``cast`` and
``description`` are accepted when present and synthesized as empty strings
when absent, and they are not used as features unless the source provided them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("cast", "description")
ALLOWED_TYPES: frozenset[str] = frozenset({"Movie", "TV Show"})


class SchemaError(ValueError):
    """Raised when a dataframe violates the catalogue contract."""


def validate_raw_frame(df: pd.DataFrame) -> None:
    """Raise ``SchemaError`` with an actionable message on contract violations."""
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise SchemaError("Catalogue is empty. Provide at least one row.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError(
            "Catalogue is missing required columns: "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". Optional columns (used only when the source file contains them): "
            + ", ".join(OPTIONAL_COLUMNS)
            + "."
        )

    type_series = df["type"].astype(str).str.strip()
    unexpected = sorted(set(type_series.unique()) - ALLOWED_TYPES)
    if unexpected:
        raise SchemaError(
            "Column 'type' must contain only 'Movie' and 'TV Show'. "
            f"Unexpected values: {unexpected}."
        )

    blank_id = df["show_id"].isna() | (df["show_id"].astype(str).str.strip() == "")
    if bool(blank_id.any()):
        count = int(blank_id.sum())
        raise SchemaError(f"Column 'show_id' has {count} blank values. Every row needs an id.")


def synthesized_columns(df: pd.DataFrame) -> list[str]:
    """Return optional columns that were added because the source omitted them."""
    stored: Any = df.attrs.get("synthesized_columns", [])
    if isinstance(stored, (list, tuple)):
        return [str(column) for column in stored]
    return []
