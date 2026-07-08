# ─────────────────────────────────────────────────────────────────────────────
# DataWhisperer — Production Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): Install Python deps into a venv
#   Stage 2 (runtime): Copy venv + app code into slim image
#
# The Streamlit app connects to Ollama over the network.
# Ollama runs as a SEPARATE service (see docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building numpy/scipy wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install deps
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Application directory
WORKDIR /app

# Copy application code (respects .dockerignore)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Create writable directories for data
RUN mkdir -p data uploads exports charts logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.port=8501"]
