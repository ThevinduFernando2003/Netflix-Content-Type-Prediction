# Decisions

Each line is a choice that was not obvious from the spec, and why it was made.

- The Excel extract has 8,790 rows and ten columns. `cast` and `description` are absent, so they are optional in the schema and are not used as features on this file. The public Kaggle file's synopsis assumption does not apply here. The real file wins.
- Excel had already coerced a handful of titles (`3%`, `23:59`, `9-Feb`, `15-Aug`, `22-Jul`, `1-Oct`, `2,215`). The exporter writes the displayed text from the cell's number format so those titles stay text.
- Missing values in this extract are the token `Not Given`, not nulls. Cleaning maps that token, and `Unknown`, to an empty string so it cannot become a learned category.
- `date_added` in the file is `M/D/YYYY`, not `September 25, 2021`. The parser tries `%m/%d/%Y`, then `%B %d, %Y`, then `%b %d, %Y`, and coerces failures. It does not infer a format per row.
- `duration_unit` is a perfect separator (6,126 movies are minutes, 2,664 TV shows are seasons, ROC-AUC 1.00). It is banned from the honest model and used only by the leaky baseline, which is not served.
- Parsed minutes, parsed seasons, and a missingness flag on minutes are the same leak. Banned.
- Every TV-format genre name, including the 16 rows whose only genre is `TV Shows`, is a perfect label proxy (flag ROC-AUC 1.00). The honest model receives `n_genres` only. That count's own ROC-AUC is 0.54, so the count stays.
- `rating` is quarantined and allowed. Its one-column ROC-AUC is 0.64. `TV-MA` and `TV-14` appear on both classes. `features.use_rating` is true.
- `has_director` is banned even though its ROC-AUC is 0.90 rather than 0.97. Single-feature accuracy is 0.95 and Cramér's V is 0.89, because `Not Given` was used for most series and almost no films. Keeping it would make the model a missing-credit detector. `features.use_has_director` is false.
- `primary_country` (AUC 0.70), `release_year`, `years_to_platform` (AUC 0.62), `n_countries`, and `title_word_count` stay. None of them is a separator.
- Description length and cast count are constant on this file because those columns were synthesized. They are omitted. A later file that actually contains them will pick them up.
- There is no synopsis, so the no-TV-vocabulary ablation strips the 20-word format list from the title, which is the only text field.
- Naive Bayes cannot accept `StandardScaler` output or a negative `years_to_platform`. ComplementNB is fit on title TF-IDF and one-hot categoricals only. That is the classic text baseline, and it is the reason the shared numeric scaler is not forced on every model.
- Tree models run `TruncatedSVD` inside the fold. A dense 5,000-d TF-IDF matrix is the wrong input for a forest on this size of data, and the SVD is fit per fold so it is not a leak.
- SMOTE is not used. Class imbalance is handled by comparing `class_weight="balanced"` with an unweighted logistic regression, and by selecting on ROC-AUC and F1-macro rather than accuracy.
- The positive class is TV Show. Movie is 0.
- The 15% test split is cut once with seed 42 before any search. The threshold is chosen on out-of-fold isotonic-calibrated probabilities from that training split. The test set is scored only in `scripts/evaluate.py`.
- A linear model within 0.005 ROC-AUC of the best non-linear model is preferred, so the API can show coefficient contributions.
- The served probability is an isotonic `CalibratedClassifierCV(method="isotonic", cv=5, ensemble=False)`. `ensemble=False` still fits the map on 5-fold training scores, then predicts with one pipeline. The default ensemble averaged five full pipelines and the local single-row p95 was 340 ms, over the 150 ms budget.
- Histogram gradient boosting won model selection (CV ROC-AUC 0.8901 versus 0.8738 for unweighted logistic regression, a gap larger than 0.005). It has no coefficient vector. Token coefficients in the evaluation report are refit from `logreg_plain` on the training split only, and they are labelled as that linear model. The API explanation list is empty for the tree, which is stated in the UI.
- Trained artifacts under `models/` are committed. The image copies them and does not see `data/raw` (it is dockerignored), and a clean clone has to serve predictions without a 50-minute retrain. `oof_probabilities.npz` stays gitignored.
- `date_added` is optional on the API, beyond the spec's request body, because `year_added`, `month_added`, and `years_to_platform` cannot be computed without it. Omitted dates are imputed inside the pipeline and the missingness indicator turns on.
- The API returns 422 when a caller sends `duration`, including a CSV that contains that column. A batch item that contains it fails that item only.
- Confidence is the distance of calibrated P(TV Show) from the tuned threshold: high at or above 0.20, medium at or above 0.08, otherwise low.
