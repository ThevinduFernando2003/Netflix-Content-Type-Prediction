"""Full-catalogue leakage audit. Association study, not a model-selection score."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_SEED, repo_root, resolve_path
from src.data.clean import clean_catalogue
from src.data.load import load_raw_csv
from src.data.split import encode_label
from src.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AuditRow:
    feature: str
    kind: str
    nmi: float
    cramers_v: float
    stump_accuracy: float
    logistic_accuracy: float
    logistic_auc: float
    note: str


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = float(chi2_contingency(table)[0])
    n_obs = float(table.to_numpy().sum())
    rank = min(table.shape) - 1
    if n_obs == 0 or rank == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n_obs * rank)))


def _discrete(series: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        labels = pd.Series("(missing)", index=series.index, dtype=object)
        observed = numeric.notna()
        if int(observed.sum()) >= 10:
            try:
                labels.loc[observed] = pd.qcut(
                    numeric.loc[observed], q=10, duplicates="drop"
                ).astype(str)
            except ValueError:
                labels.loc[observed] = numeric.loc[observed].round(3).astype(str)
        elif int(observed.sum()) > 0:
            labels.loc[observed] = numeric.loc[observed].round(3).astype(str)
        return labels.astype(str)
    return series.fillna("(missing)").astype(str).replace("", "(missing)")


def _logistic_scores(feature: pd.Series, y: np.ndarray, kind: str) -> tuple[float, float, float]:
    """In-sample stump plus 5-fold out-of-fold logistic scores for one column.

    Numeric columns keep a missingness indicator. Median-filling ``duration_minutes``
    would hide the leak the audit exists to catch.
    """
    if kind == "numeric":
        values = pd.to_numeric(feature, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_SEED),
                ),
            ]
        )
        stump_matrix = SimpleImputer(strategy="median", add_indicator=True).fit_transform(values)
    else:
        text = (
            feature.fillna("(missing)")
            .astype(str)
            .replace("", "(missing)")
            .to_numpy()
            .reshape(-1, 1)
        )
        values = text
        model = Pipeline(
            steps=[
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_SEED),
                ),
            ]
        )
        stump_matrix = pd.factorize(text.ravel())[0].reshape(-1, 1)

    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
    stump.fit(stump_matrix, y)
    stump_acc = float(accuracy_score(y, stump.predict(stump_matrix)))

    if len(np.unique(y)) < 2:
        return stump_acc, float("nan"), float("nan")
    try:
        proba = cross_val_predict(model, values, y, cv=5, method="predict_proba", n_jobs=1)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y, proba))
        acc = float(accuracy_score(y, pred))
    except Exception as exc:  # noqa: BLE001 - one bad column should not abort the audit
        logger.warning("Single-feature model failed: %s", exc)
        return stump_acc, float("nan"), float("nan")
    return stump_acc, acc, auc


def _audit_feature(
    name: str, feature: pd.Series, y_text: pd.Series, y: np.ndarray, kind: str, note: str
) -> AuditRow:
    discrete = _discrete(feature, kind)
    nmi = float(
        normalized_mutual_info_score(
            y_text.astype(str), discrete.astype(str), average_method="arithmetic"
        )
    )
    v_stat = _cramers_v(discrete, y_text.astype(str))
    stump_acc, log_acc, log_auc = _logistic_scores(feature, y, kind)
    return AuditRow(name, kind, nmi, v_stat, stump_acc, log_acc, log_auc, note)


def _crosstab_block(feature: pd.Series, y: pd.Series, kind: str, limit: int = 8) -> str:
    discrete = _discrete(feature, kind)
    table = pd.crosstab(discrete, y, margins=False)
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False).head(limit)
    headers = ["value", *[str(column) for column in table.columns]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for index, row in table.iterrows():
        cells = [str(index).replace("|", "\\|")] + [str(int(value)) for value in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.4f}"


def build_report(cleaned: pd.DataFrame) -> str:
    y_text = cleaned["type"].astype(str)
    y = encode_label(y_text).to_numpy()
    specs: list[tuple[str, pd.Series, str, str]] = [
        (
            "duration (raw string)",
            cleaned["duration"],
            "categorical",
            "Perfect separator candidate.",
        ),
        ("duration_unit", cleaned["duration_unit"], "categorical", "min versus season."),
        ("duration_minutes", cleaned["duration_minutes"], "numeric", "Null pattern is the label."),
        ("duration_seasons", cleaned["duration_seasons"], "numeric", "Null pattern is the label."),
        (
            "duration_minutes_is_null",
            cleaned["duration_minutes"].isna().astype(int),
            "categorical",
            "Missingness restates the label.",
        ),
        (
            "genre_has_banned_tv_token",
            cleaned["genre_has_banned_tv_token"],
            "categorical",
            "Genre name contains a banned TV-format token.",
        ),
        (
            "rating",
            cleaned["rating"],
            "categorical",
            "Partial leak. Quarantined pending this table.",
        ),
        (
            "has_director",
            cleaned["has_director"],
            "categorical",
            "Films list a director; series often do not.",
        ),
        (
            "primary_country",
            cleaned["primary_country"],
            "categorical",
            "Allowed categorical, audited anyway.",
        ),
        (
            "country_is_missing",
            cleaned["primary_country"].isna().astype(int),
            "categorical",
            "Missingness of country.",
        ),
        ("n_genres", cleaned["n_genres"], "numeric", "Count only. Names are banned."),
        ("n_countries", cleaned["n_countries"], "numeric", "Allowed numeric."),
        ("release_year", cleaned["release_year"], "numeric", "Allowed numeric."),
        ("year_added", cleaned["year_added"], "numeric", "Allowed numeric."),
        ("month_added", cleaned["month_added"], "numeric", "Allowed numeric."),
        (
            "years_to_platform",
            cleaned["years_to_platform"],
            "numeric",
            "year_added - release_year.",
        ),
        ("title_word_count", cleaned["title_word_count"], "numeric", "Allowed numeric."),
        (
            "description_word_count",
            cleaned["description_word_count"],
            "numeric",
            "Zero when the source has no synopsis.",
        ),
        ("n_cast", cleaned["n_cast"], "numeric", "Zero when the source has no cast column."),
        (
            "title_has_format_word",
            cleaned["title"]
            .str.contains(r"\b(series|season|episodes?|movie|film)\b", case=False, regex=True)
            .astype(int),
            "categorical",
            "Soft cue inside the title. Allowed, ablated in the no-TV-vocabulary track.",
        ),
    ]

    rows = [
        _audit_feature(name, feature, y_text, y, kind, note) for name, feature, kind, note in specs
    ]
    table_lines = [
        "| feature | kind | NMI | Cramer's V | stump accuracy | logistic accuracy | logistic ROC-AUC | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    row.feature,
                    row.kind,
                    _fmt(row.nmi),
                    _fmt(row.cramers_v),
                    _fmt(row.stump_accuracy),
                    _fmt(row.logistic_accuracy),
                    _fmt(row.logistic_auc),
                    row.note,
                ]
            )
            + " |"
        )

    crosstabs = [
        "### duration_unit",
        _crosstab_block(cleaned["duration_unit"], y_text, "categorical"),
        "",
        "### rating",
        _crosstab_block(cleaned["rating"], y_text, "categorical"),
        "",
        "### has_director (1 = a director is listed)",
        _crosstab_block(cleaned["has_director"], y_text, "categorical"),
        "",
        "### genre_has_banned_tv_token",
        _crosstab_block(cleaned["genre_has_banned_tv_token"], y_text, "categorical"),
        "",
        "### n_genres",
        _crosstab_block(cleaned["n_genres"], y_text, "numeric"),
    ]

    by_name = {row.feature: row for row in rows}
    lines = [
        "# Leakage audit",
        "",
        "Generated by `scripts/leakage_audit.py` on the full cleaned catalogue.",
        "Normalised mutual information uses `normalized_mutual_info_score` (arithmetic mean).",
        "Cramer's V is computed on the same discrete version of the feature (quantile bins for numerics).",
        "Logistic accuracy and ROC-AUC are **5-fold out-of-fold** scores of a one-column model.",
        "The decision-stump accuracy is in-sample, because a depth-1 tree is the association diagnostic.",
        "A single feature with ROC-AUC above about 0.97 is treated as a leak.",
        "",
        "## Scores",
        "",
        "\n".join(table_lines),
        "",
        "## Crosstabs",
        "",
        "\n".join(crosstabs),
        "",
        "## Verdicts",
        "",
        _verdict_prose(by_name),
        "",
    ]
    return "\n".join(lines)


def _verdict_prose(rows: dict[str, AuditRow]) -> str:
    def auc(name: str) -> str:
        return _fmt(rows[name].logistic_auc)

    rating_auc = rows["rating"].logistic_auc
    director = rows["has_director"]
    rating_decision = "BANNED" if rating_auc >= 0.97 else "ALLOWED after quarantine"
    director_is_proxy = (
        director.logistic_auc >= 0.85
        or director.logistic_accuracy >= 0.90
        or director.cramers_v >= 0.80
    )
    if director_is_proxy:
        director_decision = (
            "BANNED. Single-feature accuracy is "
            f"{_fmt(director.logistic_accuracy)} and Cramer's V is {_fmt(director.cramers_v)}. "
            "ROC-AUC can sit under the 0.97 perfect-separator line because the feature is binary "
            "and the classes are unbalanced, but the decision rule is still a catalogue-missingness "
            "detector: series usually have no director credit and films usually do. "
            "`features.use_has_director` is false."
        )
    else:
        director_decision = (
            "ALLOWED. It stays under the accuracy, AUC, and Cramer's V bars set for this column. "
            "`features.use_has_director` is true."
        )

    rating_sentence = (
        f"- `rating` logistic ROC-AUC is {auc('rating')}: **{rating_decision}**. "
        "`TV-Y`, `TV-Y7`, `TV-Y7-FV`, `TV-G`, `NR`, and `UR` skew toward TV, while `TV-MA` and "
        "`TV-14` appear on both classes. The 0.97 single-feature line is the ban threshold."
    )
    return "\n".join(
        [
            "### Banned",
            "",
            f"- `duration` raw string, logistic ROC-AUC {auc('duration (raw string)')}. "
            "`90 min` is a movie and `2 Seasons` is a TV show. The string is the label.",
            f"- `duration_unit`, logistic ROC-AUC {auc('duration_unit')}. "
            "This is the same fact, parsed. The leaky baseline is the only model allowed to use it, "
            "and that model is not served.",
            f"- `duration_minutes` (AUC {auc('duration_minutes')}) and `duration_seasons` "
            f"(AUC {auc('duration_seasons')}). Which of the two is non-null is the label. "
            "The numeric audit keeps a missingness indicator so imputation cannot hide that.",
            f"- `duration_minutes_is_null`, AUC {auc('duration_minutes_is_null')}. "
            "A missingness flag that restates the unit.",
            f"- Banned TV-format genre tokens, flag AUC {auc('genre_has_banned_tv_token')}. "
            "Names such as `TV Dramas`, `Docuseries`, and `International TV Shows` are catalogue labels, "
            "not content. The honest model receives `n_genres` only.",
            "",
            "### Quarantine decisions",
            "",
            rating_sentence,
            f"- `has_director` logistic ROC-AUC is {auc('has_director')}. {director_decision} "
            "This extract uses the placeholder `Not Given` for most series and almost no films. "
            "Keeping that placeholder as a token would be the second-order leak the spec warns about, "
            "so cleaning maps it to missing and the boolean is then judged on its own AUC.",
            "",
            "### Allowed, with the ablation",
            "",
            f"- Title text, including format words. The boolean `title_has_format_word` has AUC "
            f"{auc('title_has_format_word')}. The no-TV-vocabulary track strips a fixed 20-word stoplist "
            "before TF-IDF so the reader can see how much of the score was the word `series` or `movie`. "
            "This extract has no synopsis column, so that ablation runs on the title.",
            f"- `n_genres` AUC {auc('n_genres')}. The count is kept when it is far from a separator. "
            "The names are not used either way.",
            f"- `primary_country` AUC {auc('primary_country')}, `release_year` AUC {auc('release_year')}, "
            f"`years_to_platform` AUC {auc('years_to_platform')}.",
            f"- `description_word_count` AUC {auc('description_word_count')} and `n_cast` AUC {auc('n_cast')}. "
            "On this extract both columns were absent from the Excel file, so the features are constant "
            "and are not passed to the honest model. A file that actually contains them will pick them up.",
            "",
            "### What the honest model is allowed to see",
            "",
            "`title` (TF-IDF), `primary_country`, "
            + ("`rating`, " if rating_auc < 0.97 else "")
            + "`release_year`, `year_added`, `month_added`, "
            "`years_to_platform`, `n_genres`, `n_countries`, `title_word_count`"
            + (", `has_director`" if not director_is_proxy else "")
            + ". Nothing derived from `duration`. No genre name.",
        ]
    )


def main() -> None:
    configure_logging()
    cleaned = clean_catalogue(load_raw_csv())
    report = build_report(cleaned)
    destination = resolve_path("reports/leakage_audit.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Wrote {destination.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
