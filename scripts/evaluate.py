"""Score the untouched 15% test set and write ``reports/evaluation.md``."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from src.config import RANDOM_SEED, repo_root, resolve_path
from src.data.clean import clean_catalogue
from src.data.load import load_raw_csv
from src.data.split import encode_label, stratified_split
from src.evaluation.explain import (
    column_permutation,
    hypothesis_for_row,
    permutation_importance_top,
    top_coefficients,
)
from src.evaluation.metrics import (
    confusion,
    score_predictions,
    segment_table,
    text_report,
    threshold_sweep,
)
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_learning_curve,
    plot_roc_pr,
    plot_threshold,
)
from src.features.engineer import honest_spec, leaky_spec, prepare_features
from src.logging_conf import configure_logging, get_logger
from src.models.persistence import load_joblib, load_metadata, write_metadata
from src.models.registry import load_runtime
from src.models.tune import ModelCandidate, make_pipeline

logger = get_logger(__name__)


def _positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    return np.clip(np.asarray(probabilities)[:, classes.index(1)], 0.0, 1.0)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _latency(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    single = frame.iloc[[0]]
    for _ in range(10):
        model.predict_proba(single)
    single_times = []
    for _ in range(40):
        started = time.perf_counter()
        model.predict_proba(single)
        single_times.append((time.perf_counter() - started) * 1000)
    batch = frame.sample(n=min(1000, len(frame)), replace=True, random_state=RANDOM_SEED)
    for _ in range(3):
        model.predict_proba(batch)
    batch_times = []
    for _ in range(10):
        started = time.perf_counter()
        model.predict_proba(batch)
        batch_times.append((time.perf_counter() - started) * 1000)
    return {
        "single_p50_ms": float(np.percentile(single_times, 50)),
        "single_p95_ms": float(np.percentile(single_times, 95)),
        "batch_p50_ms": float(np.percentile(batch_times, 50)),
        "batch_p95_ms": float(np.percentile(batch_times, 95)),
        "batch_rows": float(len(batch)),
    }


def _linear_explanation_model(runtime: Any, spec: Any, x_train: pd.DataFrame, y_train: Any) -> Any:
    """Return a model with coefficients.

    The served tree has none. ``logreg_plain`` is refit on the training split
    with the C chosen during cross-validation, and it is used only for the
    token tables.
    """
    classifier = runtime.model.named_steps["clf"]
    if hasattr(classifier, "coef_") and hasattr(classifier, "predict_proba"):
        return runtime.model
    from sklearn.linear_model import LogisticRegression

    stored = runtime.metadata.get("cv_scores", {}).get("logreg_plain", {})
    params = dict(stored.get("best_params") or {})
    candidate = ModelCandidate(
        name="logreg_plain",
        estimator=LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
    )
    pipeline = make_pipeline(spec, candidate)
    if params:
        pipeline.set_params(**params)
    pipeline.fit(x_train, y_train)
    return pipeline


def _year_bucket(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "unknown"
    year = int(number)
    if year < 2000:
        return "before 2000"
    if year < 2010:
        return "2000-2009"
    if year < 2016:
        return "2010-2015"
    if year < 2019:
        return "2016-2018"
    return "2019 and later"


def _pattern_paragraph(errors: pd.DataFrame) -> str:
    count = len(errors)
    tv_missed = int((errors["true_type"] == "TV Show").sum())
    films_called_series = count - tv_missed
    banned = float(errors["genre_has_banned_tv_token"].astype(float).mean()) if count else 0.0
    no_director = float((errors["has_director"].astype(float) == 0).mean()) if count else 0.0
    short = float(errors["title"].astype(str).str.split().map(len).le(2).mean()) if count else 0.0
    return "\n".join(
        [
            f"Of the {count} most confident mistakes, {tv_missed} are TV shows scored as movies "
            f"and {films_called_series} are movies scored as TV shows.",
            f"{banned:.0%} of them carry a banned TV-format genre token in `listed_in`. "
            "The honest model never sees that token, so a series whose title reads like a film title "
            "is scored from country, rating, year, and title wording alone.",
            f"{no_director:.0%} have no director credit. That missingness is a strong catalogue cue "
            "and it was banned on purpose, which is why several series with a blank director credit "
            "are still called movies.",
            f"{short:.0%} have a title of one or two words, which leaves the TF-IDF block with almost no signal. "
            "The source file also has no synopsis, so there is no second text field to recover from. "
            "The recurring pattern is catalogue convention (genre name, missing director) that the honest "
            "model refuses to use, plus short titles that do not say whether the work is a film or a series.",
        ]
    )


def build_evaluation() -> str:
    """Score saved models on the held-out split and write the report."""
    runtime = load_runtime()
    if runtime is None:
        raise FileNotFoundError("Train the model before evaluation.")
    cleaned = clean_catalogue(load_raw_csv())
    split = stratified_split(cleaned)
    spec = honest_spec(cleaned)
    x_test = prepare_features(split.x_test, spec)
    y_test = encode_label(split.y_test).to_numpy()
    y_train = encode_label(split.y_train)
    x_train = prepare_features(split.x_train, spec)

    raw_score = _positive_probability(runtime.model, x_test)
    calibrated_score = _positive_probability(runtime.calibrator, x_test)
    threshold = runtime.threshold
    predicted = (calibrated_score >= threshold).astype(int)
    predicted_half = (calibrated_score >= 0.5).astype(int)
    if float(roc_auc_from(y_test, calibrated_score)) >= 0.999:
        raise RuntimeError(
            "Test ROC-AUC is at least 0.999. An undetected leak is still in the model."
        )

    leaky_model = load_joblib("leaky_model.joblib")
    no_tv_model = load_joblib("no_tv_model.joblib")
    leaky_score = _positive_probability(
        leaky_model, prepare_features(split.x_test, leaky_spec(cleaned))
    )
    no_tv_score = _positive_probability(no_tv_model, x_test)
    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
    dummy.fit(x_train, y_train)
    dummy_score = _positive_probability(dummy, x_test)

    honest_tuned = score_predictions(y_test, predicted, calibrated_score)
    honest_half = score_predictions(y_test, predicted_half, calibrated_score)
    leaky_metrics = score_predictions(y_test, (leaky_score >= 0.5).astype(int), leaky_score)
    no_tv_metrics = score_predictions(y_test, (no_tv_score >= 0.5).astype(int), no_tv_score)
    matrix = confusion(y_test, predicted)
    sweep = threshold_sweep(y_test, calibrated_score)
    latency = _latency(runtime.calibrator, x_test)

    figures = resolve_path("reports/figures")
    plot_confusion(
        np.array(matrix["raw"]), np.array(matrix["normalised"]), figures / "confusion_matrix.png"
    )
    plot_roc_pr(
        y_test, calibrated_score, dummy_score, figures / "roc_curve.png", figures / "pr_curve.png"
    )
    plot_calibration(y_test, raw_score, calibrated_score, figures / "calibration.png")
    plot_threshold(sweep, figures / "threshold_sweep.png")
    plot_learning_curve(
        clone(runtime.model), x_train, y_train.to_numpy(), figures / "learning_curve.png"
    )

    linear_model = _linear_explanation_model(runtime, spec, x_train, y_train)
    token_importance = permutation_importance_top(linear_model, x_test, y_test)
    if token_importance:
        plot_importance(
            [str(row["feature"]) for row in token_importance],
            [float(row["auc_drop"]) for row in token_importance],
            figures / "permutation_importance.png",
        )
    column_importance = column_permutation(runtime.calibrator, x_test, y_test)
    coefficients = top_coefficients(linear_model)
    linear_c = (
        runtime.metadata.get("cv_scores", {})
        .get("logreg_plain", {})
        .get("best_params", {})
        .get("clf__C")
    )

    detail = split.x_test.copy()
    detail["true_type"] = split.y_test.to_numpy()
    detail["predicted_type"] = np.where(predicted == 1, "TV Show", "Movie")
    detail["probability_tv_show"] = calibrated_score
    detail["confidence"] = np.where(predicted == 1, calibrated_score, 1.0 - calibrated_score)
    mistakes = (
        detail.loc[detail["true_type"] != detail["predicted_type"]]
        .sort_values("confidence", ascending=False)
        .head(25)
    )
    mistake_rows = []
    for _, series in mistakes.iterrows():
        mistake_rows.append(
            {
                "title": series["title"],
                "true_type": series["true_type"],
                "predicted_type": series["predicted_type"],
                "probability_tv_show": float(series["probability_tv_show"]),
                "hypothesis": hypothesis_for_row(series, str(series["predicted_type"])),
            }
        )

    overall_accuracy = honest_tuned["accuracy"]
    year_frame = split.x_test.copy()
    year_frame["year_bucket"] = year_frame["release_year"].map(_year_bucket)
    year_segments = segment_table(
        year_frame, y_test, predicted, "year_bucket", overall=overall_accuracy
    )
    country_counts = split.x_test["primary_country"].fillna("(missing)").value_counts()
    top_countries = set(country_counts.head(10).index)
    country_frame = split.x_test.copy()
    country_frame["country_segment"] = (
        country_frame["primary_country"]
        .fillna("(missing)")
        .map(lambda value: value if value in top_countries else "other")
    )
    country_segments = segment_table(
        country_frame, y_test, predicted, "country_segment", overall=overall_accuracy
    )
    description_note = (
        "Description length is not a segment: the Excel extract has no synopsis column, "
        "so every row has length 0."
        if "description" in set(cleaned.attrs.get("synthesized_columns", []))
        else "Description length quartiles use the synopsis character count."
    )
    length_segments: list[dict[str, Any]] = []
    if "description" not in set(cleaned.attrs.get("synthesized_columns", [])):
        length_frame = split.x_test.copy()
        length_frame["length_quartile"] = pd.qcut(
            length_frame["description_char_len"], q=4, duplicates="drop"
        ).astype(str)
        length_segments = segment_table(
            length_frame, y_test, predicted, "length_quartile", overall=overall_accuracy
        )

    brier_before = score_predictions(y_test, predicted_half, raw_score)["brier"]
    brier_after = honest_tuned["brier"]
    metadata = load_metadata()
    metadata["test_scores"] = {
        "honest_tuned_threshold": honest_tuned,
        "honest_threshold_0_5": honest_half,
        "leaky": leaky_metrics,
        "no_tv_vocabulary": no_tv_metrics,
        "brier_before_calibration": brier_before,
        "brier_after_calibration": brier_after,
        "latency": latency,
        "confusion_matrix": matrix,
    }
    write_metadata(metadata)

    comparison_rows = [
        {
            "track": "leaky baseline (duration_unit, threshold 0.5)",
            **_public_metrics(leaky_metrics),
        },
        {"track": "honest model (threshold 0.5)", **_public_metrics(honest_half)},
        {
            "track": f"honest model (tuned threshold {threshold:.2f})",
            **_public_metrics(honest_tuned),
        },
        {"track": "no-TV-vocabulary ablation (threshold 0.5)", **_public_metrics(no_tv_metrics)},
    ]
    flagged = [row for row in year_segments + country_segments + length_segments if row["flagged"]]
    flag_text = (
        "Flagged segments (accuracy at least 0.08 below the overall accuracy, support at least 20): "
        + ", ".join(
            f"{row['segment']} ({row['accuracy']:.3f}, n={row['support']})" for row in flagged
        )
        if flagged
        else "No segment with support of at least 20 sits more than 0.08 below the overall accuracy."
    )
    lines = [
        "# Evaluation",
        "",
        "Generated by `scripts/evaluate.py` on the stratified 15% test set held out before training.",
        f"Served model: `{metadata['model_name']}`. Operating threshold: **{threshold:.2f}** "
        "on calibrated P(TV Show).",
        "",
        "## Three tracks",
        "",
        "The leaky baseline includes `duration_unit`. The honest model does not. "
        "The no-TV-vocabulary model is the honest model with a fixed 20-word format stoplist "
        "removed from the title before TF-IDF. This extract has no synopsis, so the ablation "
        "is on the title. The gap between leaky and honest is the value of the banned shortcut. "
        "The gap between honest and no-TV-vocabulary is how much of the remaining score is the "
        "word `series`, `movie`, or `season` rather than the rest of the title.",
        "",
        _markdown_table(
            comparison_rows,
            [
                "track",
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "mcc",
                "roc_auc",
                "average_precision",
            ],
        ),
        "",
        "## Minority class first",
        "",
        f"TV Show precision {honest_tuned['precision_tv']:.4f}, recall {honest_tuned['recall_tv']:.4f}, "
        f"F1 {honest_tuned['f1_tv']:.4f}.",
        f"Movie precision {honest_tuned['precision_movie']:.4f}, recall {honest_tuned['recall_movie']:.4f}, "
        f"F1 {honest_tuned['f1_movie']:.4f}.",
        "",
        "```",
        text_report(y_test, predicted),
        "```",
        "",
        "## Headline metrics at the tuned threshold",
        "",
        f"- Accuracy: {honest_tuned['accuracy']:.4f}",
        f"- Balanced accuracy: {honest_tuned['balanced_accuracy']:.4f}",
        f"- F1-macro: {honest_tuned['f1_macro']:.4f}",
        f"- MCC: {honest_tuned['mcc']:.4f}",
        f"- Cohen's kappa: {honest_tuned['cohen_kappa']:.4f}",
        f"- ROC-AUC: {honest_tuned['roc_auc']:.4f}",
        f"- Average precision: {honest_tuned['average_precision']:.4f}",
        "",
        "Accuracy alone is the wrong headline. The training dummy already lands near the movie base rate.",
        "",
        "## Confusion matrix",
        "",
        "Rows are the true type and columns are the prediction, ordered Movie then TV Show.",
        "",
        f"Raw counts: {matrix['raw']}.",
        f"Row-normalised: {matrix['normalised']}.",
        "",
        "A true movie predicted as a movie is a correct film catalogue call. "
        "A true movie predicted as a TV show is a film the model would file as a series. "
        "A true TV show predicted as a TV show is a correct series call, and this is the cell the "
        "minority class has to earn. A true TV show predicted as a movie is a series the model "
        "would hide in the film list. The figure is `reports/figures/confusion_matrix.png`.",
        "",
        "## Calibration",
        "",
        f"Test Brier score before calibration: **{brier_before:.4f}**. "
        f"After isotonic calibration: **{brier_after:.4f}**.",
        "The reliability diagram is `reports/figures/calibration.png`. "
        "The API serves the calibrated probability.",
        "",
        "## Threshold sweep",
        "",
        _markdown_table(
            sweep, ["threshold", "precision_tv", "recall_tv", "f1_tv", "f1_macro", "accuracy"]
        ),
        "",
        f"The operating threshold {threshold:.2f} was chosen on training out-of-fold scores, not on this table. "
        f"High-precision (95% TV precision) threshold stored in metadata: "
        f"{metadata.get('high_precision_threshold')}.",
        "",
        "## Learning curve",
        "",
        "Train versus validation ROC-AUC as the training set grows. "
        "The curve uses 3-fold CV on the training split only (`reports/figures/learning_curve.png`). "
        "A validation score that is still rising at the right edge means more catalogue rows would help. "
        "A flat validation score means the ceiling is the features, not the sample size.",
        "",
        "## Permutation importance and coefficients",
        "",
        "Input-column permutation on the test set, using the calibrated model:",
        "",
        _markdown_table(column_importance, ["feature", "auc_drop"]),
        "",
        "Token permutation on the test set, using `logreg_plain` "
        f"(C={linear_c}) refit on the training split. The served model is "
        f"`{metadata['model_name']}` and has no coefficient vector, so these "
        "tokens describe the linear model, not the tree:",
        "",
        _markdown_table(
            token_importance or [{"feature": "(model has no coef_)", "auc_drop": 0.0}],
            ["feature", "auc_drop"],
        ),
        "",
        "Largest coefficients toward TV Show:",
        "",
        _markdown_table(
            coefficients["tv_show"] or [{"feature": "n/a", "coefficient": 0.0}],
            ["feature", "coefficient"],
        ),
        "",
        "Largest coefficients toward Movie:",
        "",
        _markdown_table(
            coefficients["movie"] or [{"feature": "n/a", "coefficient": 0.0}],
            ["feature", "coefficient"],
        ),
        "",
        "## Error analysis",
        "",
        "The 25 most confident mistakes on the test set. Confidence is the probability of the predicted class.",
        "",
        _markdown_table(
            mistake_rows,
            ["title", "true_type", "predicted_type", "probability_tv_show", "hypothesis"],
        ),
        "",
        _pattern_paragraph(mistakes),
        "",
        "## Segments",
        "",
        description_note,
        "",
        "### Release year",
        "",
        _markdown_table(year_segments, ["segment", "support", "accuracy", "flagged"]),
        "",
        "### Primary country (top 10 plus other)",
        "",
        _markdown_table(country_segments, ["segment", "support", "accuracy", "flagged"]),
        "",
    ]
    if length_segments:
        lines.extend(
            [
                "### Description length quartile",
                "",
                _markdown_table(length_segments, ["segment", "support", "accuracy", "flagged"]),
                "",
            ]
        )
    lines.extend(
        [
            flag_text,
            "",
            "## Latency",
            "",
            f"Single-row p50 {latency['single_p50_ms']:.2f} ms, p95 {latency['single_p95_ms']:.2f} ms.",
            f"Batch of {int(latency['batch_rows'])} rows: p50 {latency['batch_p50_ms']:.2f} ms, "
            f"p95 {latency['batch_p95_ms']:.2f} ms.",
            "Measured with the calibrated model that the API serves, after a short warmup.",
            "",
        ]
    )
    return "\n".join(lines)


def roc_auc_from(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _public_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc", "average_precision"]
    return {key: metrics[key] for key in keys}


def main() -> None:
    configure_logging()
    report = build_evaluation()
    destination = resolve_path("reports/evaluation.md")
    destination.write_text(report, encoding="utf-8")
    print(f"Wrote {destination.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
