# ─────────────────────────────────────────────────────────────────────────────
# SentimentScribe — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Build-time metadata
LABEL org.opencontainers.image.title="SentimentScribe API"
LABEL org.opencontainers.image.description="Book review sentiment analysis service"
LABEL org.opencontainers.image.source="https://github.com/Shah-king/SentimentScribe"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY train.py .

# Copy trained artifacts if they exist (pre-trained deployment)
# In CI/CD these are typically mounted as a volume or downloaded
COPY experiments/ ./experiments/ 2>/dev/null || true

# Ensure reports and logs directories exist
RUN mkdir -p reports logs

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
