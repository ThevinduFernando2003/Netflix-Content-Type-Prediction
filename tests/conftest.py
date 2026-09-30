"""Shared fixtures. The API client is backed by a model fit only on the mini catalogue."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RANDOM_SEED  # noqa: E402
from src.data.clean import clean_catalogue  # noqa: E402
from src.data.load import load_raw_csv  # noqa: E402
from src.data.split import encode_label, stratified_split  # noqa: E402
from src.features.engineer import honest_spec, prepare_features  # noqa: E402
from src.models.calibrate import fit_calibrator  # noqa: E402
from src.models.persistence import save_artifacts  # noqa: E402
from src.models.train import banned_feature_list  # noqa: E402
from src.models.tune import ModelCandidate, make_pipeline  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mini_netflix.csv"


@pytest.fixture(scope="session")
def catalogue():
    """Cleaned mini catalogue, including the duplicate show_id."""
    return clean_catalogue(load_raw_csv(FIXTURE))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real fitted pipeline small enough for the API tests."""
    directory = tmp_path_factory.mktemp("models")
    cleaned = clean_catalogue(load_raw_csv(FIXTURE))
    split = stratified_split(cleaned, test_size=0.25)
    spec = honest_spec(cleaned)
    x_train = prepare_features(split.x_train, spec)
    y_train = encode_label(split.y_train)
    candidate = ModelCandidate(
        name="logreg_balanced",
        estimator=LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
    )
    pipeline = make_pipeline(spec, candidate)
    pipeline.fit(x_train, y_train)
    calibrator = fit_calibrator(pipeline, x_train, y_train)
    save_artifacts(
        best_model=pipeline,
        calibrator=calibrator,
        label_encoder={"Movie": 0, "TV Show": 1},
        metadata={
            "model_name": "logreg_balanced",
            "model_version": "test",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "git_sha": "test",
            "threshold": 0.45,
            "confidence_margins": {"high": 0.20, "medium": 0.08},
            "spec": {
                "text": list(spec.text_columns),
                "categorical": list(spec.categorical_columns),
                "numeric": list(spec.numeric_columns),
                "extra_categorical": list(spec.extra_categorical),
                "strip_tv_vocabulary": False,
            },
            "banned_features": banned_feature_list(),
            "feature_columns": list(spec.all_columns),
            "positive_class": "TV Show",
            "test_scores": {
                "honest_tuned_threshold": {
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.75,
                    "f1_macro": 0.74,
                    "precision_tv": 0.7,
                    "recall_tv": 0.6,
                    "f1_tv": 0.65,
                    "roc_auc": 0.9,
                    "mcc": 0.4,
                },
                "confusion_matrix": {
                    "raw": [[10, 1], [2, 4]],
                    "normalised": [[0.9, 0.1], [0.3, 0.7]],
                },
            },
            "leakage_audit_md": "duration is banned.",
            "description_in_model": "description" in spec.text_columns,
            "use_rating": True,
            "use_has_director": False,
        },
        directory=directory,
    )
    return directory


@pytest.fixture(scope="session")
def client(model_dir: Path) -> TestClient:
    """HTTP client with rate limiting off and the mini model loaded."""
    import os

    os.environ["RATE_LIMIT_ENABLED"] = "false"
    from src.api.main import create_app

    application = create_app(model_dir)
    with TestClient(application) as test_client:
        yield test_client


def sample_payload() -> dict[str, object]:
    """A valid prediction body."""
    return {
        "title": "Harbor Heist",
        "description": "A heist crew returns to the harbor for one more job tonight.",
        "release_year": 2019,
        "country": "United States",
        "rating": "PG-13",
        "n_genres": 2,
        "n_cast": 4,
        "has_director": True,
        "date_added": "9/25/2021",
    }
