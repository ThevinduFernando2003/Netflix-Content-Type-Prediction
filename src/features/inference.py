"""Build one model row from an API payload."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.clean import _parse_dates


def build_inference_frame(fields: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    """Create the column set the fitted pipeline expects.

    Date parts stay missing when the caller does not send ``date_added``. The
    imputer inside the pipeline fills them and sets the missingness indicator.
    """
    spec = metadata["spec"]
    columns = [
        *spec["text"],
        *spec["categorical"],
        *spec["numeric"],
        *spec["extra_categorical"],
    ]
    row: dict[str, Any] = {column: np.nan for column in columns}
    title = str(fields.get("title") or "")
    description = str(fields.get("description") or "")
    country = str(fields.get("country") or "")
    country_parts = [part.strip() for part in country.split(",") if part.strip()]

    if "title" in row:
        row["title"] = title
    if "description" in row:
        row["description"] = description
    if "primary_country" in row:
        row["primary_country"] = country_parts[0] if country_parts else np.nan
    if "rating" in row:
        rating = fields.get("rating")
        row["rating"] = str(rating) if rating else np.nan
    if "n_countries" in row:
        row["n_countries"] = float(len(country_parts))
    if "n_genres" in row and fields.get("n_genres") is not None:
        row["n_genres"] = float(fields["n_genres"])
    if "n_cast" in row and fields.get("n_cast") is not None:
        row["n_cast"] = float(fields["n_cast"])
    if "has_director" in row and fields.get("has_director") is not None:
        row["has_director"] = 1.0 if fields["has_director"] else 0.0
    if "title_word_count" in row:
        row["title_word_count"] = float(len(title.split())) if title else 0.0
    if "description_word_count" in row:
        row["description_word_count"] = float(len(description.split())) if description else 0.0
    if "description_char_len" in row:
        row["description_char_len"] = float(len(description))
    if "release_year" in row:
        row["release_year"] = float(fields["release_year"])

    parsed = _parse_dates(pd.Series([fields.get("date_added") or ""])).iloc[0]
    if pd.notna(parsed):
        if "year_added" in row:
            row["year_added"] = float(parsed.year)
        if "month_added" in row:
            row["month_added"] = float(parsed.month)
        if "years_to_platform" in row:
            row["years_to_platform"] = float(parsed.year) - float(fields["release_year"])

    return pd.DataFrame([row], columns=columns)
