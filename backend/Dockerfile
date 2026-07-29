# Slim image: the engine has no heavyweight ML dependencies, so no compiler
# toolchain or model download is needed at build time.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so application edits do not invalidate the layer.
COPY requirements.txt requirements-postgres.txt ./
# Postgres driver included so the same image serves both SQLite and Postgres.
RUN pip install --no-cache-dir -r requirements-postgres.txt

COPY app ./app

# Run as a non-root user, and give it ownership of the data directory so the
# local storage backend works out of the box.
RUN useradd --create-home --uid 10001 callsense \
    && mkdir -p /data \
    && chown -R callsense:callsense /app /data
USER callsense

ENV DATABASE_URL=sqlite+aiosqlite:////data/callsense.db \
    STORAGE_LOCAL_PATH=/data/blobs

EXPOSE 8000

# Readiness, not liveness: this checks that dependencies are actually reachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/ready').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
