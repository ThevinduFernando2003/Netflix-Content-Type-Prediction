"""API helpers: confidence buckets and the shared duration rejection."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

DURATION_DETAIL = (
    "duration is excluded by design — it trivially determines the label. "
    "See /meta for the banned-feature list."
)


def reject_duration(payload: Any) -> None:
    """Raise 422 when a single object contains a duration field."""
    if isinstance(payload, dict) and "duration" in payload:
        raise HTTPException(status_code=422, detail=DURATION_DETAIL)


def confidence_label(probability_tv: float, threshold: float, high: float, medium: float) -> str:
    """Map distance from the operating threshold into low, medium, or high."""
    distance = abs(probability_tv - threshold)
    if distance >= high:
        return "high"
    if distance >= medium:
        return "medium"
    return "low"
