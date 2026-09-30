"""Cleaning rules for the known catalogue defects.

Missing text becomes an empty string. The literal ``"Not Given"`` (and similar
placeholders already present in this extract) is treated as missing so it cannot
become a token that proxies the label.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import MISSING_TOKENS, get_config
from src.data.schema import OPTIONAL_COLUMNS, synthesized_columns, validate_raw_frame
from src.logging_conf import get_logger

logger = get_logger(__name__)

_RATING_DURATION_RE = re.compile(r"^\d+\s*min$", re.IGNORECASE)
_MINUTES_RE = re.compile(r"^(\d+)\s*min$", re.IGNORECASE)
_SEASONS_RE = re.compile(r"^(\d+)\s*seasons?$", re.IGNORECASE)
_DATE_FORMATS: tuple[str, ...] = ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y")


def _is_missing_token(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def _clean_text(value: Any) -> str:
    if _is_missing_token(value):
        return ""
    return str(value).strip()


def _split_list(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part and part.lower() not in MISSING_TOKENS]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse ``date_added`` with explicit formats only. Failures stay as NaT."""
    text = series.map(_clean_text)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = text != ""
    for fmt in _DATE_FORMATS:
        if not bool(remaining.any()):
            break
        attempt = pd.to_datetime(text.where(remaining), format=fmt, errors="coerce")
        success = attempt.notna()
        parsed = parsed.where(~success, attempt)
        remaining = remaining & attempt.isna()
    return parsed


def clean_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules and return a feature-ready frame.

    The returned frame still contains ``duration`` so the leakage audit and the
    deliberate leaky baseline can see it. The honest pipeline must not request it.
    """
    validate_raw_frame(df)
    synthesized = list(synthesized_columns(df))
    out = df.copy()

    for column in OPTIONAL_COLUMNS:
        if column not in out.columns:
            out[column] = ""
            if column not in synthesized:
                synthesized.append(column)

    out = out.drop_duplicates(subset=["show_id"], keep="first").reset_index(drop=True)

    for column in (
        "title",
        "director",
        "cast",
        "country",
        "rating",
        "duration",
        "listed_in",
        "description",
    ):
        out[column] = out[column].map(_clean_text)

    shifted = out["rating"].map(lambda value: bool(_RATING_DURATION_RE.match(value)))
    n_shifted = int(shifted.sum())
    if n_shifted:
        out.loc[shifted, "duration"] = out.loc[shifted, "rating"]
        out.loc[shifted, "rating"] = ""
        logger.info("Moved %s duration-like rating values into duration.", n_shifted)

    out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")

    minutes: list[float] = []
    seasons: list[float] = []
    units: list[str] = []
    duration_failures = 0
    for raw in out["duration"].tolist():
        minute_match = _MINUTES_RE.match(raw)
        season_match = _SEASONS_RE.match(raw)
        if minute_match:
            minutes.append(float(minute_match.group(1)))
            seasons.append(np.nan)
            units.append("min")
        elif season_match:
            minutes.append(np.nan)
            seasons.append(float(season_match.group(1)))
            units.append("season")
        else:
            minutes.append(np.nan)
            seasons.append(np.nan)
            units.append("unparsed")
            if raw:
                duration_failures += 1
    out["duration_minutes"] = minutes
    out["duration_seasons"] = seasons
    out["duration_unit"] = units
    out["duration_parse_failed"] = [
        unit == "unparsed" and bool(raw) for unit, raw in zip(units, out["duration"], strict=True)
    ]

    added = _parse_dates(out["date_added"])
    out["date_added_parsed"] = added
    out["year_added"] = added.dt.year.astype("float")
    out["month_added"] = added.dt.month.astype("float")
    out["years_to_platform"] = out["year_added"] - out["release_year"]

    genre_lists = out["listed_in"].map(_split_list)
    country_lists = out["country"].map(_split_list)
    cast_lists = out["cast"].map(_split_list)
    out["genres"] = genre_lists
    out["countries"] = country_lists
    out["cast_list"] = cast_lists
    out["n_genres"] = genre_lists.map(len).astype(float)
    out["n_countries"] = country_lists.map(len).astype(float)
    out["n_cast"] = cast_lists.map(len).astype(float)
    out["primary_country"] = country_lists.map(lambda items: items[0] if items else np.nan)
    out["has_director"] = out["director"].map(lambda value: 0.0 if value == "" else 1.0)

    out["description_char_len"] = out["description"].map(len).astype(float)
    out["description_word_count"] = (
        out["description"].map(lambda text: len(text.split()) if text else 0).astype(float)
    )
    out["title_word_count"] = (
        out["title"].map(lambda text: len(text.split()) if text else 0).astype(float)
    )

    banned_genres = {item.casefold() for item in get_config()["leakage"]["banned_tv_genres"]}
    out["genre_has_banned_tv_token"] = genre_lists.map(
        lambda items: float(any(item.casefold() in banned_genres for item in items))
    )

    out.attrs["synthesized_columns"] = synthesized
    out.attrs["rating_shifts"] = n_shifted
    out.attrs["duration_parse_failures"] = duration_failures
    out.attrs["date_parse_failures"] = int(added.isna().sum())
    logger.info(
        "Cleaned catalogue: %s rows, %s rating shifts, %s duration parse failures, "
        "%s date parse failures.",
        len(out),
        n_shifted,
        duration_failures,
        int(added.isna().sum()),
    )
    return out
