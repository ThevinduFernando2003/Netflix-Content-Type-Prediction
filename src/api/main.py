"""FastAPI application. The leaky baseline is not served."""

import io
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.deps import DURATION_DETAIL, confidence_label, reject_duration
from src.api.schemas import (
    BatchItemResult,
    BatchResponse,
    FeatureContribution,
    PredictRequest,
    PredictResponse,
)
from src.config import get_config
from src.evaluation.explain import linear_contributions
from src.features.inference import build_inference_frame
from src.logging_conf import configure_logging, get_logger
from src.models.registry import Runtime, load_runtime

logger = get_logger(__name__)


def _counter(name: str, documentation: str, labels: list[str]) -> Counter:
    try:
        return Counter(name, documentation, labels)
    except ValueError:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]


def _histogram(name: str, documentation: str, buckets: tuple[float, ...]) -> Histogram:
    try:
        return Histogram(name, documentation, buckets=buckets)
    except ValueError:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]


REQUESTS = _counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
PREDICTION_LATENCY = _histogram(
    "prediction_latency_ms",
    "Single prediction latency in milliseconds",
    (5, 10, 25, 50, 100, 150, 250, 500, 1000),
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and write one access log line."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_id=%s unhandled before response", request_id)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
        logger.info(
            "request_id=%s method=%s path=%s status=%s took_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def _rate_limit() -> str:
    env = os.environ.get("RATE_LIMIT")
    if env:
        return env
    return str(get_config()["api"]["rate_limit"])


def _limiter() -> Limiter:
    enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"
    return Limiter(key_func=get_remote_address, enabled=enabled)


def _score(runtime: Runtime, payload: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    frame = build_inference_frame(payload.model_dump(), runtime.metadata)
    probability = float(
        runtime.calibrator.predict_proba(frame)[:, list(runtime.calibrator.classes_).index(1)][0]
    )
    probability = min(1.0, max(0.0, probability))
    threshold = runtime.threshold
    prediction = "TV Show" if probability >= threshold else "Movie"
    margins = runtime.metadata["confidence_margins"]
    contributions = [
        FeatureContribution(**item) for item in linear_contributions(runtime.model, frame, limit=8)
    ]
    elapsed = (time.perf_counter() - started) * 1000
    PREDICTION_LATENCY.observe(elapsed)
    return PredictResponse(
        prediction=prediction,  # type: ignore[arg-type]
        probability_movie=1.0 - probability,
        probability_tv_show=probability,
        confidence=confidence_label(probability, threshold, float(margins["high"]), float(margins["medium"])),  # type: ignore[arg-type]
        threshold_used=threshold,
        model_version=runtime.version,
        top_contributing_features=contributions,
        took_ms=round(elapsed, 3),
    )


def _require_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return runtime


def create_app(model_directory: Path | None = None) -> FastAPI:
    """Build the app. Tests pass a temporary model directory."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.runtime = load_runtime(model_directory)
        if app.state.runtime is not None:
            meta = app.state.runtime.metadata
            logger.info(
                "Serving %s version=%s git_sha=%s threshold=%s banned=%s",
                meta.get("model_name"),
                meta.get("model_version"),
                meta.get("git_sha"),
                meta.get("threshold"),
                meta.get("banned_features"),
            )
        else:
            logger.warning("Starting without a model. /ready will return 503.")
        yield

    app = FastAPI(
        title="Netflix content type classifier",
        version="1.0.0",
        summary="Predict Movie vs TV Show without the duration shortcut.",
        lifespan=lifespan,
    )
    limiter = _limiter()
    app.state.limiter = limiter
    origins = [
        item.strip()
        for item in os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        del exc
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        request_id = getattr(request.state, "request_id", "unknown")
        logging.getLogger("src.api").exception("request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error", "request_id": request_id},
        )

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness. This does not check the model."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request) -> JSONResponse:
        """Readiness. 503 until artifacts have been loaded."""
        if getattr(request.app.state, "runtime", None) is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.get("/meta", tags=["ops"])
    async def meta(request: Request) -> dict[str, Any]:
        """Model card: version, metrics, threshold, features, and the banned list."""
        runtime = _require_runtime(request)
        public = dict(runtime.metadata)
        public.pop("feature_names", None)
        public["feature_name_count"] = len(runtime.metadata.get("feature_names", []))
        return public

    @app.post("/predict", response_model=PredictResponse, tags=["predict"])
    @limiter.limit(_rate_limit())
    async def predict(request: Request) -> PredictResponse:
        """Score one title. A `duration` field is a 422, on purpose."""
        payload = await _json_body(request)
        reject_duration(payload)
        runtime = _require_runtime(request)
        return _score(runtime, _validate(payload))

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
    @limiter.limit(_rate_limit())
    async def predict_batch(request: Request) -> BatchResponse:
        """Score up to 500 items. Item errors stay inside the response."""
        payload = await _json_body(request)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise HTTPException(
                status_code=422, detail="Body must be an object with an items list."
            )
        items = payload["items"]
        limit = int(get_config()["api"]["max_batch"])
        if not 1 <= len(items) <= limit:
            raise HTTPException(
                status_code=422, detail=f"Batch size must be between 1 and {limit}."
            )
        runtime = _require_runtime(request)
        results: list[BatchItemResult] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                results.append(
                    BatchItemResult(index=index, ok=False, error="Item must be an object.")
                )
                continue
            if "duration" in item:
                results.append(BatchItemResult(index=index, ok=False, error=DURATION_DETAIL))
                continue
            try:
                parsed = PredictRequest.model_validate(item)
            except ValidationError as exc:
                results.append(
                    BatchItemResult(index=index, ok=False, error=_validation_message(exc))
                )
                continue
            results.append(BatchItemResult(index=index, ok=True, result=_score(runtime, parsed)))
        return BatchResponse(results=results)

    @app.post("/predict/csv", tags=["predict"])
    @limiter.limit(_rate_limit())
    async def predict_csv(request: Request, file: UploadFile = File(...)) -> Response:
        """Upload a CSV and receive a CSV of predictions. A duration column is rejected."""
        runtime = _require_runtime(request)
        raw = await file.read()
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not parse the CSV: {exc}") from exc
        if "duration" in frame.columns:
            raise HTTPException(status_code=422, detail=DURATION_DETAIL)
        limit = int(get_config()["api"]["max_batch"])
        if len(frame) > limit:
            raise HTTPException(status_code=422, detail=f"CSV row limit is {limit}.")
        output_rows: list[dict[str, Any]] = []
        for index, record in enumerate(frame.to_dict(orient="records")):
            if "duration" in record:
                output_rows.append({"row": index, "error": DURATION_DETAIL})
                continue
            try:
                parsed = PredictRequest.model_validate(_csv_record(record))
            except ValidationError as exc:
                output_rows.append({"row": index, "error": _validation_message(exc)})
                continue
            scored = _score(runtime, parsed)
            output_rows.append({"row": index, "error": "", **scored.model_dump()})
        buffer = io.StringIO()
        pd.DataFrame(output_rows).to_csv(buffer, index=False)
        return Response(content=buffer.getvalue(), media_type="text/csv")

    @app.post("/explain", response_model=PredictResponse, tags=["predict"])
    @limiter.limit(_rate_limit())
    async def explain(request: Request) -> PredictResponse:
        """Same prediction as /predict, including per-feature contributions."""
        payload = await _json_body(request)
        reject_duration(payload)
        runtime = _require_runtime(request)
        return _score(runtime, _validate(payload))

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        """Prometheus counters and the prediction latency histogram."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="Request body must be JSON.") from exc


def _validate(payload: Any) -> PredictRequest:
    try:
        return PredictRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_message(exc)) from exc


def _validation_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", []))
    return f"{location}: {error.get('msg', 'invalid')}"


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if value is None or str(value).strip() == "":
            continue
        if key in {"release_year", "n_genres", "n_cast"}:
            cleaned[key] = int(float(value))
        elif key == "has_director":
            cleaned[key] = str(value).strip().lower() in {"1", "true", "yes"}
        else:
            cleaned[key] = value
    return cleaned


app = create_app()
