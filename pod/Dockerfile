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

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision \
    transformers>=4.45.0 \
    accelerate \
    qwen-vl-utils \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    opencv-python-headless \
    av \
    pillow \
    einops \
    pynvml \
    redis \
    aiofiles \
    tqdm

# Pre-download Qwen3-VL-8B-Instruct (October 15, 2025)
RUN python3 -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
    Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', \
    device_map='auto', trust_remote_code=True, torch_dtype='bfloat16', cache_dir='/models'); \
    AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True, cache_dir='/models')"

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
