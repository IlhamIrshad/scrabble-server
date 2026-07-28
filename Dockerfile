FROM python:3.12-slim

# libgl/libglib for OpenCV; fonts-dejavu-core provides the default template font.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/scrabble-debug \
    && chown -R appuser /tmp/scrabble-debug /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBUG_DIR=/tmp/scrabble-debug \
    TEMPLATE_CACHE=/tmp/scrabble-templates.npz

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
