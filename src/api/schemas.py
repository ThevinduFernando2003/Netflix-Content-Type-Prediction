"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureContribution(BaseModel):
    """One local contribution. Positive direction means the feature pushes toward TV Show."""

    feature: str
    contribution: float
    direction: Literal["movie", "tv_show"]


class PredictRequest(BaseModel):
    """One catalogue row. ``duration`` is rejected before this model sees the body."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "The Last Horizon",
                    "description": "A retired pilot attempts one final flight across the Andes.",
                    "release_year": 2019,
                    "country": "United States",
                    "rating": "PG-13",
                    "n_genres": 2,
                    "n_cast": 4,
                    "has_director": True,
                    "date_added": "9/25/2021",
                }
            ]
        }
    )

    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=10, max_length=5000)
    release_year: int = Field(ge=1900, le=2100)
    country: str | None = None
    rating: str | None = None
    n_genres: int | None = Field(default=None, ge=0, le=20)
    n_cast: int | None = Field(default=None, ge=0, le=100)
    has_director: bool | None = None
    date_added: str | None = Field(
        default=None,
        description="Optional platform date, for example 9/25/2021. Omitted dates are imputed.",
    )


class PredictResponse(BaseModel):
    """Calibrated prediction for one row."""

    prediction: Literal["Movie", "TV Show"]
    probability_movie: float = Field(ge=0, le=1)
    probability_tv_show: float = Field(ge=0, le=1)
    confidence: Literal["low", "medium", "high"]
    threshold_used: float
    model_version: str
    top_contributing_features: list[FeatureContribution]
    took_ms: float


class BatchItemResult(BaseModel):
    """One slot in a batch response. Failed items do not fail the rest of the batch."""

    index: int
    ok: bool
    result: PredictResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Batch predictions in input order."""

    results: list[BatchItemResult]


class MediaResponse(BaseModel):
    """Wikipedia poster and synopsis for display. Never a model input."""

    found: bool
    query: str
    title: str
    description: str | None = None
    image_url: str | None = None
    page_url: str | None = None
    source: str
    content_type: Literal["Movie", "TV Show"] | None = None
