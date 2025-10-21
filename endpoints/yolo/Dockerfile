FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
RUN pip3 install --no-cache-dir \
    runpod \
    ultralytics \
    pillow \
    torch \
    torchvision

# Download YOLOv11m model (will be cached in image)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# Copy handler
COPY handler.py .

# Start the handler
CMD ["python3", "handler.py"]
