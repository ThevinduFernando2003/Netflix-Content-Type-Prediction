"""HTTP contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import sample_payload


def test_health_ready_and_meta(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200
    meta = client.get("/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["model_version"] == "test"
    assert "duration" in " ".join(body["banned_features"])
    assert body["threshold"] == 0.45
    assert "feature_names" not in body


def test_predict_and_explain_shapes(client: TestClient) -> None:
    for path in ("/predict", "/explain"):
        response = client.post(path, json=sample_payload())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"] in {"Movie", "TV Show"}
        assert abs(body["probability_movie"] + body["probability_tv_show"] - 1) < 1e-6
        assert body["confidence"] in {"low", "medium", "high"}
        assert body["threshold_used"] == 0.45
        assert isinstance(body["top_contributing_features"], list)
        assert "X-Request-ID" in response.headers


def test_validation_errors_name_the_field(client: TestClient) -> None:
    missing = sample_payload()
    del missing["title"]
    response = client.post("/predict", json=missing)
    assert response.status_code == 422
    assert "title" in response.json()["detail"]

    invalid = sample_payload()
    invalid["release_year"] = 1800
    assert client.post("/predict", json=invalid).status_code == 422

    short = sample_payload()
    short["description"] = "too short"
    assert client.post("/predict", json=short).status_code == 422

    huge = sample_payload()
    huge["description"] = "word " * 2000
    assert client.post("/predict", json=huge).status_code == 422


def test_adversarial_inputs_do_not_crash(client: TestClient) -> None:
    cases = [
        {
            **sample_payload(),
            "title": "🎬",
            "description": "A film about a lighthouse and a storm.",
        },
        {
            **sample_payload(),
            "description": "Robert'); DROP TABLE titles;-- is not a synopsis really",
        },
        {
            **sample_payload(),
            "title": "../../etc/passwd",
            "description": "A film about a stolen password file.",
        },
        {**sample_payload(), "description": "the and a of to in on at by or"},
    ]
    for payload in cases:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text


def test_batch_limits_and_partial_errors(client: TestClient) -> None:
    good = sample_payload()
    bad = {"title": "X"}
    leaked = {**good, "duration": "90 min"}
    response = client.post("/predict/batch", json={"items": [good, bad, leaked]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["index"] for item in results] == [0, 1, 2]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert results[2]["ok"] is False
    assert "duration" in results[2]["error"]

    too_many = client.post("/predict/batch", json={"items": [good] * 501})
    assert too_many.status_code == 422
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_csv_upload(client: TestClient) -> None:
    happy = (
        "title,description,release_year,country,rating\n"
        "Harbor Heist,A heist crew returns to the harbor tonight,2019,United States,PG-13\n"
    )
    response = client.post("/predict/csv", files={"file": ("rows.csv", happy, "text/csv")})
    assert response.status_code == 200
    assert "prediction" in response.text
    assert "text/csv" in response.headers["content-type"]

    malformed = client.post("/predict/csv", files={"file": ("bad.csv", b"", "text/csv")})
    assert malformed.status_code == 422

    leaked = client.post(
        "/predict/csv",
        files={"file": ("leak.csv", "title,duration\nA,90 min\n", "text/csv")},
    )
    assert leaked.status_code == 422
    assert "duration" in leaked.json()["detail"]


def test_metrics(client: TestClient) -> None:
    client.post("/predict", json=sample_payload())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"prediction_latency_ms" in response.content


def test_not_ready_without_artifacts(tmp_path: Path) -> None:
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    from src.api.main import create_app

    with TestClient(create_app(tmp_path)) as lonely:
        assert lonely.get("/health").status_code == 200
        assert lonely.get("/ready").status_code == 503
        missing = lonely.post("/predict", json=sample_payload())
        assert missing.status_code == 503
        leaked = lonely.post("/predict", json={**sample_payload(), "duration": "1 Season"})
        assert leaked.status_code == 422


def test_real_artifact_sanity() -> None:
    """Skip cleanly when make train has not been run."""
    from src.config import resolve_path

    if not (resolve_path("models") / "metadata.json").exists():
        pytest.skip("trained artifact is not present")
    from src.features.inference import build_inference_frame
    from src.models.registry import load_runtime

    runtime = load_runtime()
    assert runtime is not None
    film = build_inference_frame(
        {
            "title": "The Last Horizon",
            "description": "A retired pilot attempts one final flight across the Andes in this film.",
            "release_year": 2019,
            "country": "United States",
            "rating": "PG-13",
            "n_genres": 2,
            "has_director": True,
        },
        runtime.metadata,
    )
    series = build_inference_frame(
        {
            "title": "Harbor Lane Season 1",
            "description": "In this series a detective returns to the harbor each season.",
            "release_year": 2020,
            "country": "United Kingdom",
            "rating": "TV-MA",
            "n_genres": 3,
            "has_director": False,
        },
        runtime.metadata,
    )
    film_tv = float(
        runtime.calibrator.predict_proba(film)[:, list(runtime.calibrator.classes_).index(1)][0]
    )
    series_tv = float(
        runtime.calibrator.predict_proba(series)[:, list(runtime.calibrator.classes_).index(1)][0]
    )
    assert series_tv > film_tv
