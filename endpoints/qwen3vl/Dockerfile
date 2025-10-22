FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install latest Python packages for Qwen3-VL (October 2025 release)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision \
    transformers>=4.45.0 \
    accelerate \
    sentencepiece \
    pillow \
    runpod \
    qwen-vl-utils \
    av \
    einops

# Pre-download Qwen3-VL-8B-Instruct model (LATEST - October 15, 2025)
# 8B params, 256K context, 32-language OCR, FP8 checkpoints
RUN python3 -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', \
    device_map='auto', trust_remote_code=True, torch_dtype='bfloat16', cache_dir='/models')"

# Copy handler
COPY handler.py .

CMD ["python3", "handler.py"]
