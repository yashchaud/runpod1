# Optimized Dockerfile for Qwen3-VL with vLLM on RunPod
# Uses pre-built vLLM image to save space and build time

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (FFmpeg for video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Install Python packages in stages to manage disk space
# Stage 1: Core dependencies
RUN pip install --no-cache-dir \
    vllm>=0.11.0 \
    transformers>=4.47.0 \
    && rm -rf /root/.cache/pip/* /tmp/*

# Stage 2: RunPod and video processing
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    ffmpeg-python>=0.2.0 \
    opencv-python-headless>=4.10.0 \
    && rm -rf /root/.cache/pip/* /tmp/*

# Stage 3: Utilities
RUN pip install --no-cache-dir \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    aiohttp>=3.9.0 \
    psutil>=5.9.0 \
    && rm -rf /root/.cache/pip/* /tmp/*

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Set Python path
ENV PYTHONPATH=/app/src:${PYTHONPATH}

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
