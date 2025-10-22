FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch first (largest package) - split to reduce build space
RUN pip3 install --no-cache-dir torch==2.4.0 && \
    rm -rf /root/.cache/pip && \
    pip3 install --no-cache-dir torchvision && \
    rm -rf /root/.cache/pip

# Install transformers and ML packages
RUN pip3 install --no-cache-dir transformers>=4.45.0 accelerate && \
    rm -rf /root/.cache/pip && \
    pip3 install --no-cache-dir qwen-vl-utils einops && \
    rm -rf /root/.cache/pip

# Install API and processing packages
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    opencv-python-headless \
    av \
    pillow \
    pynvml \
    redis \
    aiofiles \
    tqdm && \
    rm -rf /root/.cache/pip

# Note: Model will be downloaded on first run to save Docker build space
# This reduces image build from ~30GB to ~10GB
# Model download happens automatically on pod startup

# Copy application files
COPY adaptive_processor.py .
COPY server.py .
COPY frame_sampler.py .
COPY vram_detector.py .

# Create directories
RUN mkdir -p /workspace/jobs /workspace/results /workspace/uploads

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start server
CMD ["python3", "server.py"]
