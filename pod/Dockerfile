# Use RunPod's PyTorch base image (PyTorch already installed!)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install transformers and ML packages (PyTorch already in base image!)
RUN pip3 install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate \
    qwen-vl-utils \
    einops && \
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
