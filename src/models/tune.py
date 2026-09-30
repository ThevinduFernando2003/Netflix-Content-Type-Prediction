"""Cross-validated model comparison on identical folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.config import RANDOM_SEED, get_config
from src.features.build_pipeline import build_model_pipeline
from src.features.engineer import FeatureSpec
from src.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCandidate:
    """One estimator family and the preprocessor options it needs."""

    name: str
    estimator: BaseEstimator
    param_distributions: dict[str, Any] | None = None
    drop_numeric: bool = False
    svd_components: int | None = None
    densify: bool = False


@dataclass
class CandidateReport:
    """CV scores for one candidate."""

    name: str
    best_params: dict[str, Any]
    metrics: dict[str, tuple[float, float]]
    fit_time_mean: float
    pipeline: Any
    drop_numeric: bool
    svd_components: int | None
    densify: bool
    notes: list[str] = field(default_factory=list)


def scoring_map() -> dict[str, Any]:
    """Primary metric is ROC-AUC. Per-class scores use Movie=0 and TV Show=1."""

    def _class_score(metric: Any, label: int) -> Any:
        return make_scorer(metric, pos_label=label, zero_division=0)

    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision_movie": _class_score(precision_score, 0),
        "precision_tv": _class_score(precision_score, 1),
        "recall_movie": _class_score(recall_score, 0),
        "recall_tv": _class_score(recall_score, 1),
        "f1_movie": _class_score(f1_score, 0),
        "f1_tv": _class_score(f1_score, 1),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }


def default_candidates() -> list[ModelCandidate]:
    """At least six families, including the stratified dummy and both class-weight settings."""
    svd = int(get_config()["features"]["svd_components"])
    return [
        ModelCandidate(
            name="dummy_stratified",
            estimator=DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        ),
        ModelCandidate(
            name="logreg_balanced",
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
            param_distributions={"clf__C": loguniform(1e-2, 1e2)},
        ),
        ModelCandidate(
            name="logreg_plain",
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight=None,
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
            param_distributions={"clf__C": loguniform(1e-2, 1e2)},
        ),
        ModelCandidate(
            name="complement_nb",
            estimator=ComplementNB(),
            param_distributions={"clf__alpha": loguniform(1e-2, 10.0)},
            drop_numeric=True,
        ),
        ModelCandidate(
            name="linear_svc",
            estimator=LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=4000,
                random_state=RANDOM_SEED,
            ),
            param_distributions={"clf__C": loguniform(1e-2, 1e2)},
        ),
        ModelCandidate(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            param_distributions={
                "clf__n_estimators": [80, 160, 240],
                "clf__max_depth": [8, 16, None],
                "clf__min_samples_leaf": [1, 3, 8],
            },
            svd_components=svd,
            densify=True,
        ),
        ModelCandidate(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                random_state=RANDOM_SEED,
                early_stopping=False,
            ),
            param_distributions={
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [3, 6, None],
                "clf__max_iter": [80, 150],
                "clf__min_samples_leaf": [10, 20],
            },
            svd_components=svd,
            densify=True,
        ),
    ]


def _nb_spec(spec: FeatureSpec) -> FeatureSpec:
    return FeatureSpec(
        text_columns=spec.text_columns,
        categorical_columns=spec.categorical_columns,
        numeric_columns=(),
        extra_categorical=spec.extra_categorical,
        strip_tv_vocabulary=spec.strip_tv_vocabulary,
        name=f"{spec.name}_nb",
    )


def make_pipeline(spec: FeatureSpec, candidate: ModelCandidate) -> Any:
    """Build the fold-safe pipeline for one candidate."""
    feature_spec = _nb_spec(spec) if candidate.drop_numeric else spec
    return build_model_pipeline(
        feature_spec,
        clone(candidate.estimator),
        scale_numeric=not candidate.drop_numeric,
        svd_components=candidate.svd_components,
        densify=candidate.densify,
    )


def _folds(y: pd.Series | np.ndarray) -> StratifiedKFold:
    n_splits = int(get_config()["cv"]["n_splits"])
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def _summarise(cv_result: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for key, values in cv_result.items():
        if not key.startswith("test_"):
            continue
        name = key.removeprefix("test_")
        summary[name] = (float(np.mean(values)), float(np.std(values)))
    return summary


def evaluate_candidate(
    spec: FeatureSpec,
    candidate: ModelCandidate,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int | None = None,
) -> CandidateReport:
    """Search (when the candidate has a grid) and score the winner on the same folds."""
    pipeline = make_pipeline(spec, candidate)
    folds = _folds(y_train)
    notes: list[str] = []
    best_params: dict[str, Any] = {}
    if candidate.param_distributions:
        iterations = int(get_config()["cv"]["n_iter"] if n_iter is None else n_iter)
        logger.info("Searching %s over %s iterations", candidate.name, iterations)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=candidate.param_distributions,
            n_iter=iterations,
            scoring="roc_auc",
            cv=folds,
            refit=True,
            random_state=RANDOM_SEED,
            n_jobs=1,
            error_score="raise",
            verbose=0,
        )
        search.fit(x_train, y_train)
        pipeline = search.best_estimator_
        best_params = {str(key): _jsonable(value) for key, value in search.best_params_.items()}
    else:
        logger.info("Fitting %s with stratified CV and no search", candidate.name)
        pipeline.fit(x_train, y_train)

    logger.info("Scoring %s on the shared folds", candidate.name)
    cv_result = cross_validate(
        clone(pipeline),
        x_train,
        y_train,
        cv=folds,
        scoring=scoring_map(),
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    if candidate.drop_numeric:
        notes.append("Text and one-hot features only; numeric columns can be negative.")
    if candidate.name == "linear_svc":
        notes.append(
            "ROC-AUC uses decision_function. Isotonic calibration is applied only if this model wins."
        )
    return CandidateReport(
        name=candidate.name,
        best_params=best_params,
        metrics=_summarise(cv_result),
        fit_time_mean=float(np.mean(cv_result["fit_time"])),
        pipeline=pipeline,
        drop_numeric=candidate.drop_numeric,
        svd_components=candidate.svd_components,
        densify=candidate.densify,
        notes=notes,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def select_winner(reports: list[CandidateReport]) -> CandidateReport:
    """Pick the highest ROC-AUC, preferring a linear model within 0.005 AUC."""
    ranked = sorted(
        reports,
        key=lambda report: (
            report.metrics["roc_auc"][0],
            report.metrics["f1_macro"][0],
        ),
        reverse=True,
    )
    best = ranked[0]
    if best.name == "dummy_stratified" and len(ranked) > 1:
        best = ranked[1]
    best_auc = best.metrics["roc_auc"][0]
    linear = [report for report in reports if report.name.startswith("logreg")]
    if linear:
        linear_best = max(linear, key=lambda report: report.metrics["roc_auc"][0])
        if best_auc - linear_best.metrics["roc_auc"][0] <= 0.005:
            logger.info(
                "Preferring %s (AUC %.4f) over %s (AUC %.4f) for token-level explanations.",
                linear_best.name,
                linear_best.metrics["roc_auc"][0],
                best.name,
                best_auc,
            )
            return linear_best
    return best


def assert_scores_sane(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Guard used by tests: probabilities must be finite and rankable."""
    if not np.isfinite(y_score).all():
        raise RuntimeError("Non-finite scores.")
    if len(np.unique(y_true)) > 1:
        roc_auc_score(y_true, y_score)
        average_precision_score(y_true, y_score)
