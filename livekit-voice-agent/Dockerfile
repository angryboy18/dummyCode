# Dockerfile for GCP Cloud Run deployment of LiveKit Voice Agent
# syntax=docker/dockerfile:1

# ── Stage 1: Base image with UV + Python 3.13 ──
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Prevent Python from buffering stdout/stderr (critical for Cloud Run logging)
ENV PYTHONUNBUFFERED=1

# GCP Cloud Run injects PORT env var; LiveKit agent HTTP health-check listens here
ENV PORT=8080

# Create a non-privileged user for security
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/app" \
  --shell "/sbin/nologin" \
  --uid "${UID}" \
  appuser

# Install build dependencies for native Python extensions
RUN sed -i 's/deb.debian.org/mirror.us-central1.gcp.debian.org/g' /etc/apt/sources.list.d/debian.sources || true && \
  apt-get update && apt-get install -y \
  gcc \
  g++ \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install dependencies (cached layer) ──
COPY pyproject.toml uv.lock ./
RUN mkdir -p src
RUN uv sync --locked

# ── Stage 3: Copy application code ──
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Pre-download ML models (Silero VAD, turn-detector, etc.) at build time
RUN uv run "agent.py" download-files

# Expose the health-check port for Cloud Run
EXPOSE ${PORT}

# Start the LiveKit agent worker
CMD ["uv", "run", "agent.py", "start"]
