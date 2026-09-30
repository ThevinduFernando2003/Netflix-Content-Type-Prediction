"""Streamlit client. It talks to the API and does not load the model."""

from __future__ import annotations

import io
import os

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = {
    "Obvious movie": {
        "title": "The Last Horizon",
        "description": "A retired pilot attempts one final flight across the Andes in this film.",
        "release_year": 2019,
        "country": "United States",
        "rating": "PG-13",
        "n_genres": 2,
        "n_cast": 4,
        "has_director": True,
        "date_added": "9/25/2021",
    },
    "Obvious series": {
        "title": "Harbor Lane Season 1",
        "description": "In this series a detective returns to the harbor each season to finish the case.",
        "release_year": 2020,
        "country": "United Kingdom",
        "rating": "TV-MA",
        "n_genres": 3,
        "n_cast": 8,
        "has_director": False,
        "date_added": "9/24/2021",
    },
    "Ambiguous case": {
        "title": "Night Archive",
        "description": "An archivist reconstructs a disappeared expedition from tapes, letters, and silence.",
        "release_year": 2018,
        "country": "Canada",
        "rating": "TV-14",
        "n_genres": 2,
        "n_cast": 3,
        "has_director": True,
        "date_added": "",
    },
}


def _error(message: str) -> None:
    st.error(message)


def _get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{API_BASE_URL}{path}", timeout=20)
    except httpx.HTTPError:
        _error(f"The API at {API_BASE_URL} is not reachable. Start it, then refresh this page.")
        return None
    if response.status_code >= 400:
        _error(f"{path} returned {response.status_code}.")
        return None
    return response.json()


@st.cache_data(ttl=60)
def load_meta() -> dict | None:
    try:
        response = httpx.get(f"{API_BASE_URL}/meta", timeout=20)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def _predict(payload: dict) -> dict | None:
    try:
        with st.spinner("Scoring..."):
            response = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
    except httpx.HTTPError:
        _error(f"The API at {API_BASE_URL} is not reachable. Start it, then try again.")
        return None
    if response.status_code != 200:
        detail = (
            response.json().get("detail", response.text)
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text
        )
        _error(f"The API rejected this row ({response.status_code}): {detail}")
        return None
    return response.json()


def _form_payload() -> dict:
    title = st.text_input("Title", value=st.session_state.get("title", "The Last Horizon"))
    description = st.text_area(
        "Synopsis",
        value=st.session_state.get("description", EXAMPLES["Obvious movie"]["description"]),
        height=140,
    )
    left, right = st.columns(2)
    release_year = left.number_input(
        "Release year",
        min_value=1900,
        max_value=2100,
        value=int(st.session_state.get("release_year", 2019)),
    )
    country = right.text_input("Country", value=st.session_state.get("country", "United States"))
    rating = left.text_input("Rating", value=st.session_state.get("rating", "PG-13"))
    date_added = right.text_input(
        "Date added (optional, M/D/YYYY)", value=st.session_state.get("date_added", "9/25/2021")
    )
    n_genres = left.number_input(
        "Genre count", min_value=0, max_value=20, value=int(st.session_state.get("n_genres", 2))
    )
    n_cast = right.number_input(
        "Cast count", min_value=0, max_value=100, value=int(st.session_state.get("n_cast", 4))
    )
    has_director = st.checkbox(
        "Director is credited", value=bool(st.session_state.get("has_director", True))
    )
    return {
        "title": title,
        "description": description,
        "release_year": int(release_year),
        "country": country or None,
        "rating": rating or None,
        "n_genres": int(n_genres),
        "n_cast": int(n_cast),
        "has_director": has_director,
        "date_added": date_added or None,
    }


def _apply_example(name: str) -> None:
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


def _contribution_chart(rows: list[dict]) -> None:
    if not rows:
        st.caption(
            "The served model is a gradient-boosted tree, so this response has no token coefficients. "
            "The linear model's tokens are in the evaluation report."
        )
        return
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7, 3.2))
    colors = [
        "#c45911" if direction == "tv_show" else "#1f4e79" for direction in frame["direction"]
    ]
    axis.barh(frame["feature"], frame["contribution"], color=colors)
    axis.axvline(0, color="#6b7280", linewidth=0.8)
    axis.set_xlabel("Contribution toward TV Show (positive) or Movie (negative)")
    axis.set_title("Top contributing features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _predict_tab(meta: dict | None) -> None:
    st.subheader("Predict")
    columns = st.columns(3)
    for column, name in zip(columns, EXAMPLES, strict=True):
        if column.button(name, use_container_width=True):
            _apply_example(name)
            st.rerun()
    if meta is not None and not meta.get("description_in_model", True):
        st.info(
            "This training extract has no synopsis column, so the model scores the title and the "
            "catalogue fields. The synopsis is still sent, and it is not a model feature."
        )
    payload = _form_payload()
    if st.button("Score this title", type="primary"):
        result = _predict(payload)
        if result is None:
            return
        st.markdown(f"## {result['prediction']}")
        probability = result["probability_tv_show"]
        st.progress(
            probability, text=f"P(TV Show) = {probability:.2f} · confidence {result['confidence']}"
        )
        st.caption(
            f"P(Movie) = {result['probability_movie']:.2f}. "
            f"Threshold {result['threshold_used']:.2f}. "
            f"Model {result['model_version']}. Took {result['took_ms']:.1f} ms."
        )
        _contribution_chart(result["top_contributing_features"])


def _batch_tab() -> None:
    st.subheader("Batch")
    template = (
        "title,description,release_year,country,rating,n_genres,n_cast,has_director,date_added\n"
    )
    template += "The Last Horizon,A retired pilot attempts one final flight across the Andes in this film.,2019,United States,PG-13,2,4,true,9/25/2021\n"
    st.download_button(
        "Download a template CSV", template, file_name="template.csv", mime="text/csv"
    )
    upload = st.file_uploader("CSV of titles", type=["csv"])
    if upload is None:
        return
    try:
        with st.spinner("Uploading..."):
            response = httpx.post(
                f"{API_BASE_URL}/predict/csv",
                files={"file": (upload.name, upload.getvalue(), "text/csv")},
                timeout=60,
            )
    except httpx.HTTPError:
        _error(f"The API at {API_BASE_URL} is not reachable.")
        return
    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        _error(f"The API rejected the file ({response.status_code}): {detail}")
        return
    frame = pd.read_csv(io.StringIO(response.text))
    st.dataframe(frame, use_container_width=True)
    st.download_button(
        "Download predictions", response.text, file_name="predictions.csv", mime="text/csv"
    )


def _performance_tab(meta: dict | None) -> None:
    st.subheader("Model performance")
    if not meta or not meta.get("test_scores"):
        st.info(
            "Test metrics appear here after `make evaluate`. The API `/meta` payload does not have them yet."
        )
        return
    scores = meta["test_scores"]["honest_tuned_threshold"]
    columns = st.columns(4)
    columns[0].metric("ROC-AUC", f"{scores['roc_auc']:.3f}")
    columns[1].metric("Balanced accuracy", f"{scores['balanced_accuracy']:.3f}")
    columns[2].metric("F1-macro", f"{scores['f1_macro']:.3f}")
    columns[3].metric("MCC", f"{scores['mcc']:.3f}")
    st.caption(
        f"TV Show precision {scores.get('precision_tv', 0):.3f}, recall {scores.get('recall_tv', 0):.3f}, F1 {scores.get('f1_tv', 0):.3f}."
    )
    matrix = meta["test_scores"].get("confusion_matrix", {}).get("raw")
    if matrix:
        fig, axis = plt.subplots(figsize=(4.2, 3.4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], ["Movie", "TV Show"])
        axis.set_yticks([0, 1], ["Movie", "TV Show"])
        axis.set_xlabel("Predicted type")
        axis.set_ylabel("True type")
        axis.set_title("Test confusion matrix (counts)")
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row][col]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    brier_before = meta["test_scores"].get("brier_before_calibration")
    brier_after = meta["test_scores"].get("brier_after_calibration")
    if brier_before is not None:
        st.write(f"Brier score before calibration: {brier_before:.4f}. After: {brier_after:.4f}.")


def _leakage_tab(meta: dict | None) -> None:
    st.subheader("The leakage story")
    st.markdown(
        "A raw duration of `90 min` is a movie and `2 Seasons` is a TV show. "
        "Genre names such as `TV Dramas` and `Docuseries` are the same kind of shortcut. "
        "The honest model is not allowed to see either one. The gap between the leaky baseline "
        "and the honest model is the point of this project."
    )
    if meta and meta.get("test_scores"):
        rows = []
        for key, label in (
            ("leaky", "Leaky baseline"),
            ("honest_tuned_threshold", "Honest model"),
            ("no_tv_vocabulary", "No TV vocabulary"),
        ):
            block = meta["test_scores"].get(key)
            if block:
                rows.append(
                    {
                        "track": label,
                        "roc_auc": block["roc_auc"],
                        "accuracy": block["accuracy"],
                        "f1_macro": block["f1_macro"],
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if meta:
        st.write("Banned features:")
        st.code("\n".join(meta.get("banned_features", [])))
        audit = meta.get("leakage_audit_md") or ""
        if audit:
            st.markdown(audit)
    else:
        _error("Load the API to see the banned-feature list and the audit.")


def _about_tab(meta: dict | None) -> None:
    st.subheader("About")
    st.markdown(
        """
This classifier predicts whether a Netflix catalogue row is a Movie or a TV Show
from metadata that does not already contain the answer.

The honest model is a histogram gradient boosting classifier on title TF-IDF
(reduced inside the pipeline), primary country, rating, release year, the year and
month the title was added, years between release and the platform, genre count,
country count, and title length. Probabilities shown here are isotonic-calibrated.
The decision threshold was chosen to maximise F1-macro on out-of-fold training scores.
The tree has no token coefficients; the evaluation report shows those from the
logistic regression that was trained on the same features.

**Limitations.** The supplied catalogue has no synopsis and no cast list, so the model
cannot read the blurb a person would read. It is English-centric, it learns this
catalogue's wording, and it has no audio or video signal. If Netflix changes how
directors, ratings, or titles are filled in, the score will move. `has_director` was
measured and banned because a missing director credit already classifies about 95% of
rows.

**Next.** Train on a catalogue that actually includes synopses, with the same leakage
ban. Add a human review queue for low-confidence rows. Recalibrate when the catalogue
mix shifts.
        """
    )
    if meta:
        st.caption(
            f"Served model {meta.get('model_name')} · version {meta.get('model_version')} · "
            f"git {meta.get('git_sha', '')[:8]} · threshold {meta.get('threshold')}"
        )


def main() -> None:
    st.set_page_config(page_title="Netflix content type", layout="wide")
    st.title("Netflix content type")
    st.caption("Movie or TV Show, without using duration or TV-genre names.")
    meta = load_meta()
    if meta is None:
        _error(f"The API at {API_BASE_URL} is not reachable, so live scoring is unavailable.")
    predict, batch, performance, leakage, about = st.tabs(
        ["Predict", "Batch", "Model performance", "The leakage story", "About"]
    )
    with predict:
        _predict_tab(meta)
    with batch:
        _batch_tab()
    with performance:
        _performance_tab(meta)
    with leakage:
        _leakage_tab(meta)
    with about:
        _about_tab(meta)


main()
