# â”€â”€ Stage 1: Python dependencies + ML models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
# CPU-only torch first: the default PyPI wheel bundles CUDA and adds several GB
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt

# Bake the embedding and reranker models into the image (same names the pipeline loads)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# â”€â”€ Stage 2: Runtime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=5000

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache
# Backend code and prebuilt search indices; visitor data and secrets are excluded.
COPY . .

# A fresh repository has no local index files; build them from checked-in public documents.
RUN python main.py --mode ingest

# Files from a Windows build context arrive world-writable: make code read-only for the app user.
# The app only writes leads, conversations, SQLite, uploads, rebuilt indices and eval results.
RUN find /app -type d -exec chmod 755 {} + \
 && find /app -type f -exec chmod 644 {} + \
 && mkdir -p data documents results evaluation/results \
 && chown -R app:app data documents results evaluation/results

USER app
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '5000'), timeout=4)"

# gunicorn on $PORT: 2 workers x 4 threads by default (override with WEB_CONCURRENCY / GUNICORN_THREADS)
CMD ["python", "start.py"]

