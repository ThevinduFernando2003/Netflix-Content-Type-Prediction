"""Cleaning rules from the data contract."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.clean import clean_catalogue
from src.data.load import load_raw_csv
from src.data.schema import SchemaError, validate_raw_frame
from tests.conftest import FIXTURE


def test_rating_duration_shift(catalogue: pd.DataFrame) -> None:
    row = catalogue.loc[catalogue["show_id"] == "m1"].iloc[0]
    assert row["rating"] == ""
    assert row["duration"] == "74 min"
    assert row["duration_minutes"] == 74
    assert row["duration_unit"] == "min"


def test_missing_text_is_empty_not_unknown(catalogue: pd.DataFrame) -> None:
    no_director = catalogue.loc[catalogue["show_id"] == "m2"].iloc[0]
    placeholder = catalogue.loc[catalogue["show_id"] == "t2"].iloc[0]
    assert no_director["director"] == ""
    assert placeholder["director"] == ""
    assert no_director["has_director"] == 0
    assert placeholder["has_director"] == 0
    assert "Unknown" not in catalogue["director"].tolist()
    assert "Not Given" not in catalogue["director"].tolist()


def test_null_cast_country_rating_and_counts(catalogue: pd.DataFrame) -> None:
    empty_cast = catalogue.loc[catalogue["show_id"] == "m3"].iloc[0]
    no_country = catalogue.loc[catalogue["show_id"] == "m4"].iloc[0]
    no_rating = catalogue.loc[catalogue["show_id"] == "m5"].iloc[0]
    multi = catalogue.loc[catalogue["show_id"] == "m7"].iloc[0]
    five = catalogue.loc[catalogue["show_id"] == "m8"].iloc[0]
    assert empty_cast["n_cast"] == 0
    assert pd.isna(no_country["primary_country"])
    assert no_country["n_countries"] == 0
    assert no_rating["rating"] == ""
    assert multi["n_countries"] == 2
    assert multi["primary_country"] == "United States"
    assert five["n_genres"] == 5


def test_blank_description_accent_and_comma(catalogue: pd.DataFrame) -> None:
    blank = catalogue.loc[catalogue["show_id"] == "m9"].iloc[0]
    accented = catalogue.loc[catalogue["show_id"] == "m10"].iloc[0]
    comma = catalogue.loc[catalogue["show_id"] == "m11"].iloc[0]
    assert blank["description"] == ""
    assert blank["description_word_count"] == 0
    assert accented["title"] == "Amélie"
    assert "," in comma["title"]


def test_same_title_can_be_both_types(catalogue: pd.DataFrame) -> None:
    pair = catalogue.loc[catalogue["title"] == "The Heist"]
    assert set(pair["type"]) == {"Movie", "TV Show"}
    assert len(pair) == 2


def test_duplicate_show_id_keeps_first() -> None:
    raw = load_raw_csv(FIXTURE)
    assert int(raw["show_id"].duplicated().sum()) == 1
    cleaned = clean_catalogue(raw)
    assert cleaned["show_id"].duplicated().sum() == 0
    kept = cleaned.loc[cleaned["show_id"] == "m1"].iloc[0]
    assert kept["title"] == "Shifted Runtime"


def test_duration_units_and_explicit_dates(catalogue: pd.DataFrame) -> None:
    movie = catalogue.loc[catalogue["show_id"] == "m6"].iloc[0]
    series = catalogue.loc[catalogue["show_id"] == "t1"].iloc[0]
    shifted = catalogue.loc[catalogue["show_id"] == "m1"].iloc[0]
    assert movie["duration_unit"] == "min"
    assert series["duration_unit"] == "season"
    assert series["duration_seasons"] == 2
    assert pd.notna(shifted["date_added_parsed"])
    assert int(shifted["year_added"]) == 2021
    assert int(shifted["month_added"]) == 9
    march = catalogue.loc[catalogue["show_id"] == "t3"].iloc[0]
    assert int(march["month_added"]) == 3


def test_schema_rejects_bad_type(catalogue: pd.DataFrame) -> None:
    broken = catalogue.copy()
    broken["type"] = "Special"
    with pytest.raises(SchemaError, match="Movie"):
        validate_raw_frame(broken)


def test_schema_rejects_missing_column(catalogue: pd.DataFrame) -> None:
    broken = catalogue.drop(columns=["duration"])
    with pytest.raises(SchemaError, match="duration"):
        validate_raw_frame(broken)
