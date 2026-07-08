# ─────────────────────────────────────────────────────────────────────────────
# DataWhisperer — Production Dockerfile
#
# Multi-stage build optimized for:
#   • Minimal image size (~450MB vs ~1.8GB naive)
#   • Layer caching (deps change less often than code)
#   • Security (non-root, no build tools in runtime)
#   • Correct signal handling (tini as PID 1)
#
# Build args:
#   PYTHON_VERSION — Python base image tag (default: 3.12-slim)
#
# Usage:
#   docker build -t datawhisperer .
#   docker build --build-arg PYTHON_VERSION=3.11-slim -t datawhisperer .
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12-slim


# ── Stage 1: Dependency Builder ──────────────────────────────────────────────
FROM python:${PYTHON_VERSION} AS builder

WORKDIR /build

# System deps for compiling numpy/scipy/scikit-learn wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into an isolated venv.
# COPY requirements.txt first to maximize Docker layer cache hits —
# code changes won't invalidate the expensive pip install layer.
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Production Runtime ──────────────────────────────────────────────
FROM python:${PYTHON_VERSION} AS runtime

# ── Security: non-root user ─────────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# ── Minimal runtime system dependencies ─────────────────────────────────────
# tini: lightweight init — handles PID 1 responsibilities (signal forwarding,
#        zombie reaping) that Python cannot do correctly as PID 1.
# libopenblas0: runtime shared lib for numpy/scipy BLAS operations.
# curl: required for Docker HEALTHCHECK probes.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
        libopenblas0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy venv from builder stage ────────────────────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    # Prevent matplotlib from trying to open display
    MPLBACKEND=Agg

# ── Application directory ───────────────────────────────────────────────────
WORKDIR /app

# Copy application code (respects .dockerignore)
# Ordered from least-frequently-changed to most-frequently-changed
# to maximize layer cache hits during development.
COPY .streamlit/ ./.streamlit/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY app.py .

# ── Create writable directories & set ownership ─────────────────────────────
RUN mkdir -p data uploads exports charts logs && \
    chown -R appuser:appuser /app

# ── Switch to non-root user ─────────────────────────────────────────────────
USER appuser

# ── Streamlit runtime configuration ─────────────────────────────────────────
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint: tini ensures proper signal handling ──────────────────────────
ENTRYPOINT ["tini", "--"]
CMD ["streamlit", "run", "app.py", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
