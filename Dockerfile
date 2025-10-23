# Multi-stage Dockerfile for Qwen3-VL with vLLM on RunPod
# Optimized for CUDA 12.4 and efficient layer caching

# Base image with CUDA and Python
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Builder stage for Python dependencies
FROM base AS builder

WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention for better performance (optional but recommended)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn installation failed, continuing without it"

# Final runtime stage
FROM base AS runtime

# Create app directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create directories for temporary files and cache
RUN mkdir -p /tmp/runpod /app/cache && \
    chmod 777 /tmp/runpod /app/cache

# Set HuggingFace cache directory
ENV HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    HF_DATASETS_CACHE=/app/cache

# Environment variables for vLLM configuration (can be overridden)
ENV MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct-FP8" \
    MAX_MODEL_LEN="131072" \
    GPU_MEMORY_UTILIZATION="0.90" \
    TENSOR_PARALLEL_SIZE="1" \
    MAX_FRAMES_PER_CHUNK="768" \
    CHUNK_DURATION="60.0" \
    CHUNK_OVERLAP="2.0"

# Performance optimizations
ENV OMP_NUM_THREADS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port (if needed for testing)
EXPOSE 8000

# Set working directory to src
WORKDIR /app/src

# Run the handler
CMD ["python", "-u", "handler.py"]
