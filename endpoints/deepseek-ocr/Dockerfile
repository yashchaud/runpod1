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

# Install Python packages for DeepSeek-OCR (October 20, 2025 release)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision \
    transformers>=4.45.0 \
    accelerate \
    pillow \
    runpod \
    sentencepiece \
    einops

# Pre-download DeepSeek-OCR model (LATEST - October 20, 2025)
# 3B params, 97% accuracy, 200K pages/day throughput, 10x compression
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
    AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-ocr', \
    device_map='auto', trust_remote_code=True, torch_dtype='bfloat16', cache_dir='/models'); \
    AutoProcessor.from_pretrained('deepseek-ai/deepseek-ocr', trust_remote_code=True, cache_dir='/models')"

# Copy handler
COPY handler.py .

CMD ["python3", "handler.py"]
