"""Load the raw catalogue without letting pandas guess dtypes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_config, resolve_path
from src.data.schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, validate_raw_frame
from src.logging_conf import get_logger

logger = get_logger(__name__)


def raw_csv_path() -> Path:
    """Return the configured raw CSV path."""
    return resolve_path(get_config()["paths"]["raw_csv"])


def load_raw_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV as strings and enforce the schema.

    All columns are read as strings so titles such as ``3%``, ``23:59`` and
    ``9-Feb`` are not coerced. Optional columns missing from this extract are
    added as empty strings and recorded on ``df.attrs['synthesized_columns']``.
    """
    csv_path = path or raw_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw catalogue not found at {csv_path}. "
            "Run `python scripts/export_excel.py` if you still have the Excel source, "
            "or place netflix_titles.csv at data/raw/."
        )

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    synthesized: list[str] = []
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
            synthesized.append(column)
            logger.warning(
                "Column %s is absent from %s. It was added as an empty string and "
                "will not be used as a model feature.",
                column,
                csv_path.name,
            )

    validate_raw_frame(frame)
    frame.attrs["synthesized_columns"] = synthesized
    frame.attrs["source_path"] = str(csv_path)
    logger.info("Loaded %s rows from %s", len(frame), csv_path)
    return frame


def column_order() -> list[str]:
    """Stable raw column order used by the fixture and the exporter."""
    return [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
