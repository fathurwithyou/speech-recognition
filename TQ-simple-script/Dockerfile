# Multi-stage Dockerfile for Whisper Benchmarking
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY benchmark.py ./
COPY BENCHMARK_USAGE.md ./

# Install dependencies using uv
RUN uv sync --frozen

# Create directories for data and results
RUN mkdir -p /app/timit_eval /app/benchmark_results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose ports
EXPOSE 8765

# Default command - can be overridden
CMD ["tail", "-f", "/dev/null"]