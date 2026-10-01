# API

Base URL locally: `http://127.0.0.1:8000`. The responses below were captured from that server after `scripts/train.py` and `scripts/evaluate.py`, on 30 September 2026. OpenAPI lives at `/docs`.

## GET /health

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

## GET /ready

```bash
curl -s http://127.0.0.1:8000/ready
```

```json
{"status":"ready"}
```

Before artifacts are loaded this returns **503** and `{"status":"not_ready"}`.

## GET /meta

```bash
curl -s http://127.0.0.1:8000/meta
```

The live body is the training `metadata.json` plus `feature_name_count`, without the raw `feature_names` list. On this run that included:

- `model_name`: `hist_gradient_boosting`
- `model_version`: `1.0.0`
- `threshold`: `0.37`
- `winner_cv_roc_auc`: `0.8900584878387997`
- `row_count`: `8790`, `train_rows`: `7471`, `test_rows`: `1319`
- `feature_columns`: `title`, `primary_country`, `rating`, `release_year`, `year_added`, `month_added`, `years_to_platform`, `n_genres`, `n_countries`, `title_word_count`
- `feature_name_count`: `123`
- `description_in_model`: `false`
- `calibration_ensemble`: `false`
- `high_precision_threshold`: `null`
- `banned_features`: duration fields, `listed_in`, `genres`, `has_director`, and the TV-format genre names through `TV Shows`
- `test_scores.honest_tuned_threshold.roc_auc`: `0.8846` when rounded to four decimals (the JSON stores the full float)
- `leakage_audit_md`: the full text of `reports/leakage_audit.md` (6,533 characters in this response)

`cv_scores` for all seven models is in the same JSON.

## POST /predict

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"The Last Horizon\",\"description\":\"A retired pilot attempts one final flight across the Andes in this film.\",\"release_year\":2019,\"country\":\"United States\",\"rating\":\"PG-13\",\"n_genres\":2,\"n_cast\":4,\"has_director\":true,\"date_added\":\"9/25/2021\"}"
```

Captured response:

```json
{
  "prediction": "Movie",
  "probability_movie": 0.9983388704318937,
  "probability_tv_show": 0.0016611295681063123,
  "confidence": "high",
  "threshold_used": 0.37,
  "model_version": "1.0.0",
  "top_contributing_features": [],
  "took_ms": 42.114
}
```

`top_contributing_features` is empty because the served model is a tree and has no coefficient vector. `took_ms` moves from call to call; 42.114 ms is the value from this capture.

A body that contains `duration` returns **422**:

```json
{"detail":"duration is excluded by design — it trivially determines the label. See /meta for the banned-feature list."}
```

An empty title returns **422**:

```json
{"detail":"title: String should have at least 1 character"}
```

## POST /explain

Same body as `/predict`. This capture returned the same prediction and probabilities, with `took_ms` **29.742** and the same empty contribution list.

## POST /predict/batch

```bash
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d "{\"items\":[{\"title\":\"The Last Horizon\",\"description\":\"A retired pilot attempts one final flight across the Andes in this film.\",\"release_year\":2019,\"country\":\"United States\",\"rating\":\"PG-13\",\"n_genres\":2,\"n_cast\":4,\"has_director\":true,\"date_added\":\"9/25/2021\"},{\"title\":\"x\",\"description\":\"too short\",\"release_year\":2020}]}"
```

Captured response:

```json
{
  "results": [
    {
      "index": 0,
      "ok": true,
      "result": {
        "prediction": "Movie",
        "probability_movie": 0.9983388704318937,
        "probability_tv_show": 0.0016611295681063123,
        "confidence": "high",
        "threshold_used": 0.37,
        "model_version": "1.0.0",
        "top_contributing_features": [],
        "took_ms": 27.764
      },
      "error": null
    },
    {
      "index": 1,
      "ok": false,
      "result": null,
      "error": "description: String should have at least 10 characters"
    }
  ]
}
```

More than 500 items returns **422**. A single item that includes `duration` fails that item only.

## POST /predict/csv

Multipart field `file`. A column named `duration` returns **422** with the same detail as `/predict`. This one-row upload returned:

```text
row,error,prediction,probability_movie,probability_tv_show,confidence,threshold_used,model_version,top_contributing_features,took_ms
0,,Movie,0.9983388704318937,0.0016611295681063123,high,0.37,1.0.0,[],39.769
```

## GET /media

Poster and synopsis for display. The text is not a model feature. Source is the public Wikipedia API.

```bash
curl -s "http://127.0.0.1:8000/media?title=Inception&content_type=Movie"
```

A blank `title` returns **422**. When Wikipedia has no usable page the status is still **200** and `found` is `false`.

## GET /metrics

```bash
curl -s http://127.0.0.1:8000/metrics
```

Prometheus text. This process reported `http_requests_total` and a `prediction_latency_ms` histogram. The histogram counts depend on how many predictions the process has served.
