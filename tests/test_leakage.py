"""The leakage guard must fail loudly if a banned feature comes back."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED
from src.data.split import encode_label, stratified_split
from src.features.build_pipeline import build_model_pipeline
from src.features.engineer import honest_spec, prepare_features
from src.models.train import assert_feature_names, assert_honest_inputs
from tests.conftest import sample_payload


def test_fitted_pipeline_has_no_banned_feature(catalogue, client) -> None:
    spec = honest_spec(catalogue)
    assert_honest_inputs(spec)
    split = stratified_split(catalogue, test_size=0.25)
    pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_SEED),
    )
    pipeline.fit(prepare_features(split.x_train, spec), encode_label(split.y_train))
    names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    assert_feature_names(names)
    banned_bits = [
        "duration",
        "listed_in",
        "has_director",
        "tv dramas",
        "docuseries",
        "international tv shows",
    ]
    for name in names:
        lowered = name.casefold()
        for bit in banned_bits:
            assert bit not in lowered, name


def test_predict_rejects_duration(client) -> None:
    payload = sample_payload()
    payload["duration"] = "2 Seasons"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"].startswith("duration is excluded by design")


def test_reintroduced_duration_name_fails() -> None:
    with pytest.raises(RuntimeError, match="Banned feature"):
        assert_feature_names(["categorical__duration_unit_min"])
