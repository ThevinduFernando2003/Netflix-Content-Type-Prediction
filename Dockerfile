FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY app ./app
COPY scripts/docker_healthcheck.py ./scripts/docker_healthcheck.py
COPY config.yaml .
COPY models ./models
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    SERVICE=api
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "scripts/docker_healthcheck.py"]
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
