"""Model comparison, threshold tuning, and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED
from src.data.load import load_raw_csv
from src.data.split import encode_label, stratified_split
from src.features.engineer import honest_spec, prepare_features
from src.models.calibrate import out_of_fold_probabilities, tune_threshold
from src.models.train import run_training
from src.models.tune import CandidateReport, ModelCandidate, evaluate_candidate, select_winner
from tests.conftest import FIXTURE


def _xy(catalogue: pd.DataFrame):
    spec = honest_spec(catalogue)
    split = stratified_split(catalogue, test_size=0.25)
    return spec, prepare_features(split.x_train, spec), encode_label(split.y_train)


def test_two_fits_match(catalogue) -> None:
    spec, x_train, y_train = _xy(catalogue)
    candidate = ModelCandidate(
        name="logreg_plain",
        estimator=LogisticRegression(max_iter=300, solver="liblinear", random_state=RANDOM_SEED),
    )
    first = evaluate_candidate(spec, candidate, x_train, y_train, n_iter=1)
    second = evaluate_candidate(spec, candidate, x_train, y_train, n_iter=1)
    np.testing.assert_allclose(
        first.pipeline.predict_proba(x_train),
        second.pipeline.predict_proba(x_train),
    )
    assert first.metrics["roc_auc"] == second.metrics["roc_auc"]


def test_threshold_is_chosen_on_oof_scores(catalogue) -> None:
    spec, x_train, y_train = _xy(catalogue)
    candidate = ModelCandidate(
        name="logreg_balanced",
        estimator=LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
    )
    report = evaluate_candidate(spec, candidate, x_train, y_train, n_iter=1)
    _raw, calibrated = out_of_fold_probabilities(report.pipeline, x_train, y_train)
    tuned = tune_threshold(y_train, calibrated)
    assert 0.05 <= tuned["threshold"] <= 0.95
    assert tuned["f1_macro"] > 0


def test_linear_model_wins_a_near_tie() -> None:
    def report(name: str, auc: float) -> CandidateReport:
        return CandidateReport(
            name=name,
            best_params={},
            metrics={"roc_auc": (auc, 0.0), "f1_macro": (0.5, 0.0)},
            fit_time_mean=0.1,
            pipeline=None,
            drop_numeric=False,
            svd_components=None,
            densify=False,
        )

    winner = select_winner(
        [report("hist_gradient_boosting", 0.90), report("logreg_balanced", 0.897)]
    )
    assert winner.name == "logreg_balanced"


def test_run_training_on_the_fixture(tmp_path) -> None:
    metadata = run_training(
        candidates=[
            ModelCandidate(
                name="dummy_stratified",
                estimator=DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
            ),
            ModelCandidate(
                name="logreg_balanced",
                estimator=LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
                param_distributions={"clf__C": [0.5, 1.0]},
            ),
        ],
        n_iter=1,
        reports_path=tmp_path / "model_comparison.md",
        source=load_raw_csv(FIXTURE),
        artifact_dir=tmp_path / "models",
        strict_auc=False,
    )
    assert "threshold" in metadata
    assert (tmp_path / "models" / "metadata.json").exists()
    report = (tmp_path / "model_comparison.md").read_text(encoding="utf-8")
    assert "dummy_stratified" in report
