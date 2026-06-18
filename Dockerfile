FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Non-root user
RUN addgroup --system mercury && adduser --system --ingroup mercury mercury

# Copy installed packages from builder
COPY --from=builder /install /usr/local

COPY app ./app

USER mercury

ENV PORT=8000 \
    LOG_FORMAT=json \
    LOG_LEVEL=info \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--log-level", "warning"]
