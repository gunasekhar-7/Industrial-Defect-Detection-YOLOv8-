# Multi-stage production image for the Industrial Defect Detection API.
#
# Stage 1 (builder): resolve the pinned runtime requirements into a virtualenv.
# Stage 2 (runtime): slim base, non-root user, only the venv + application.
# Headless OpenCV removes the need for libgl1/libglib, and the healthcheck is
# Python-based, so neither curl nor any GUI library ships in the final image.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Unprivileged user; no libgl1/libglib/curl: opencv is headless and the
# healthcheck below uses the standard library only.
RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
COPY configs/ ./configs/
COPY src/ ./src/
COPY models/ ./models/

USER app

EXPOSE 8000 8501

# /health returns 503 when the model is missing; urlopen raises on any
# non-2xx status, so the container is marked unhealthy without curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
