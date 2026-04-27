# syntax=docker/dockerfile:1.7
# =====================================================================
# Multi-stage build using uv. Builds the final-state app (checkpoint-4).
# Override CHECKPOINT at build time to ship an earlier checkpoint:
#   docker build --build-arg CHECKPOINT=checkpoint-2-agent-loop .
# =====================================================================

ARG CHECKPOINT=checkpoint-4-orchestration

# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Layer cache: install deps from the lockfile before copying source.
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev || \
    uv sync --no-install-project --no-dev

ARG CHECKPOINT
COPY checkpoints/${CHECKPOINT}/app ./app
COPY data ./data
COPY README.md ./


# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/app ./app
COPY --from=builder /build/data ./data

# ChromaDB writes its index here; mount a volume in production.
RUN mkdir -p /app/.chroma && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
