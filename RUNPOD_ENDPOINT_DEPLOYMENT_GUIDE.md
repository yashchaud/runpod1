# RunPod Serverless Endpoint Deployment Guide

**Last Updated**: 2025-01-XX
**For**: Video Understanding Agent Project

---

## Overview

This guide details how to deploy 3 serverless endpoints on RunPod for our video understanding system:

1. **YOLOv11** - Object detection and tracking
2. **Faster-Whisper** - Speech transcription
3. **Qwen3-VL-8B** - Vision-language understanding

---

## Prerequisites

✅ **Completed**:
- [x] RunPod account created
- [x] RunPod API key obtained
- [x] Credits added to account ($25-50 minimum recommended)

⏳ **Before deploying**:
- [ ] API key added to `.env` file
- [ ] Understand pricing (see Cost Estimates below)

---

## ENDPOINT 1: YOLOv11 Object Detection

### Overview
- **Purpose**: Real-time object detection and person tracking
- **Model**: YOLOv11m (medium variant for balance of speed/accuracy)
- **Library**: Ultralytics
- **Expected Latency**: 50-150ms per image

### Deployment Method

**Option A: Custom Docker Image (Recommended)**

Since YOLOv11 is new (released Sept 2024), we'll create a custom worker using the Ultralytics library.

#### 1. Create Handler File

Create `endpoints/yolo/handler.py`:

```python
import runpod
from ultralytics import YOLO
import base64
import io
import json
from PIL import Image

# Load model once at startup
print("Loading YOLOv11m model...")
model = YOLO('yolo11m.pt')
print("Model loaded successfully!")

def handler(event):
    """
    Handler for YOLOv11 object detection

    Input format:
    {
        "input": {
            "image": "base64_encoded_image",
            "conf": 0.25,  # optional confidence threshold
            "iou": 0.7,    # optional IOU threshold
            "classes": []  # optional: filter by class IDs
        }
    }

    Output format:
    {
        "detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95,
                "class": 0,
                "class_name": "person"
            },
            ...
        ],
        "count": 5,
        "inference_time": 0.045
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Decode base64 image
        image_b64 = input_data.get('image')
        if not image_b64:
            return {"error": "No image provided"}

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        # Get parameters
        conf_threshold = input_data.get('conf', 0.25)
        iou_threshold = input_data.get('iou', 0.7)
        filter_classes = input_data.get('classes', None)

        # Run inference
        results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=filter_classes,
            verbose=False
        )[0]

        # Format detections
        detections = []
        for box in results.boxes:
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'confidence': float(box.conf[0].cpu().numpy()),
                'class': int(box.cls[0].cpu().numpy()),
                'class_name': model.names[int(box.cls[0])]
            })

        inference_time = time.time() - start_time

        return {
            'detections': detections,
            'count': len(detections),
            'inference_time': inference_time,
            'image_size': [results.orig_shape[1], results.orig_shape[0]]
        }

    except Exception as e:
        return {"error": str(e)}

# Start the serverless worker
runpod.serverless.start({'handler': handler})
```

#### 2. Create Dockerfile

Create `endpoints/yolo/Dockerfile`:

```dockerfile
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
    pillow

# Download YOLOv11m model (will be cached in image)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

# Copy handler
COPY handler.py .

# Start the handler
CMD ["python3", "handler.py"]
```

#### 3. Build and Push Docker Image

```bash
# Build image
docker build -t your-dockerhub-username/yolo11-runpod:latest ./endpoints/yolo

# Push to Docker Hub
docker push your-dockerhub-username/yolo11-runpod:latest
```

#### 4. Deploy on RunPod

1. Go to RunPod Console → **Serverless** → **New Endpoint**
2. Click **Custom Source** → **Docker Image**
3. Enter: `your-dockerhub-username/yolo11-runpod:latest`
4. Configure:
   - **Name**: `yolo11-detection`
   - **GPU**: RTX 4090 or A40 (good price/performance)
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 3
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 30 seconds
   - **Container Disk**: 10 GB
5. Click **Deploy**

#### 5. Test Endpoint

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Read test image
with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call endpoint
endpoint = runpod.Endpoint("ENDPOINT_ID")
result = endpoint.run_sync({
    "input": {
        "image": image_b64,
        "conf": 0.25
    }
})

print(result)
```

### Estimated Costs
- **RTX 4090**: ~$0.0004 per inference (avg 100ms)
- **A40**: ~$0.0003 per inference (slightly slower)

---

## ENDPOINT 2: Faster-Whisper Speech Transcription

### Overview
- **Purpose**: Audio transcription with timestamps
- **Model**: Faster-Whisper Large-v3-Turbo
- **Performance**: 2-4x faster than OpenAI Whisper
- **Expected Latency**: 0.5-2 seconds per 5-second audio chunk

### Deployment Method

**Option A: Use Official RunPod Template (Easiest)**

RunPod provides an official `worker-faster_whisper` template.

#### 1. Deploy via RunPod Console

1. Go to RunPod Console → **Serverless** → **New Endpoint**
2. Look for **Quick Deploy** templates
3. Search for "**faster-whisper**" or use custom deployment
4. Use Docker Image: `runpod/worker-faster_whisper:latest`

**OR**

#### 2. Deploy Custom Faster-Whisper

Create `endpoints/whisper/handler.py`:

```python
import runpod
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import numpy as np

# Load model at startup
print("Loading Faster-Whisper large-v3-turbo model...")
model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
    download_root="/models"
)
print("Model loaded successfully!")

def handler(event):
    """
    Handler for Faster-Whisper transcription

    Input format:
    {
        "input": {
            "audio": "base64_encoded_audio",
            "language": "en",  # optional, auto-detect if not provided
            "task": "transcribe",  # or "translate"
            "vad_filter": true,
            "word_timestamps": false
        }
    }

    Output format:
    {
        "transcription": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "confidence": 0.95
            },
            ...
        ],
        "language": "en",
        "duration": 10.5
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Decode audio
        audio_b64 = input_data.get('audio')
        if not audio_b64:
            return {"error": "No audio provided"}

        audio_bytes = base64.b64decode(audio_b64)
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Get parameters
        language = input_data.get('language', None)
        task = input_data.get('task', 'transcribe')
        vad_filter = input_data.get('vad_filter', True)
        word_timestamps = input_data.get('word_timestamps', False)

        # Transcribe
        segments, info = model.transcribe(
            audio,
            language=language,
            task=task,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            beam_size=5
        )

        # Format output
        transcription = []
        for segment in segments:
            transcription.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob
            })

        processing_time = time.time() - start_time

        return {
            'transcription': transcription,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'processing_time': processing_time
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({'handler': handler})
```

#### 3. Create Dockerfile

Create `endpoints/whisper/Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    runpod \
    faster-whisper \
    soundfile \
    numpy

# Pre-download model
RUN mkdir -p /models
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', download_root='/models')"

COPY handler.py .

CMD ["python3", "handler.py"]
```

#### 4. Deploy Configuration

1. **Name**: `faster-whisper-transcription`
2. **GPU**: A40 or A100 (for faster processing)
3. **Min Workers**: 0
4. **Max Workers**: 5
5. **Idle Timeout**: 5 seconds
6. **Execution Timeout**: 60 seconds
7. **Container Disk**: 15 GB

### Estimated Costs
- **A40**: ~$0.001 per minute of audio
- **A100**: ~$0.0015 per minute (faster processing)

---

## ENDPOINT 3: Qwen3-VL-8B Vision-Language Model

### Overview
- **Purpose**: Video understanding, emotion detection, context reasoning
- **Model**: Qwen3-VL-8B-Instruct
- **Framework**: vLLM (for optimized inference)
- **Expected Latency**: 1-3 seconds per inference

### Deployment Method

**Option A: Using vLLM Worker Template (Recommended)**

RunPod provides official vLLM worker support for multimodal models.

#### 1. Deploy via RunPod Console

1. Go to RunPod Console → **Serverless** → **New Endpoint**
2. Under **Quick Deploy**, find **Serverless vLLM**
3. Click **Configure**

#### 2. Configuration

Set these environment variables:

```bash
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
TOKENIZER_NAME=Qwen/Qwen3-VL-8B-Instruct
HF_TOKEN=your_huggingface_token_if_needed
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.9
DTYPE=float16
```

**Docker Image**: `runpod/worker-v1-vllm:stable-cuda12.1.0`

(Check for latest version at: https://github.com/runpod-workers/worker-vllm)

#### 3. Deployment Settings

1. **Name**: `qwen3-vl-8b-reasoning`
2. **GPU**: A100 40GB or 80GB (required for 8B VLM)
3. **Min Workers**: 0
4. **Max Workers**: 3
5. **Idle Timeout**: 10 seconds
6. **Execution Timeout**: 60 seconds
7. **Container Disk**: 25 GB

#### 4. Test Endpoint

```python
import runpod
import base64

runpod.api_key = "YOUR_API_KEY"

# Prepare image
with open("frame.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

endpoint = runpod.Endpoint("QWEN3_ENDPOINT_ID")

# Call with vision input
result = endpoint.run_sync({
    "input": {
        "prompt": "Describe what is happening in this video frame. Pay attention to people's emotions and body language.",
        "image": image_b64,
        "max_tokens": 512,
        "temperature": 0.7
    }
})

print(result)
```

**Option B: Custom Handler (More Control)**

Create `endpoints/qwen3-vl/handler.py`:

```python
import runpod
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
import base64
import io
from PIL import Image

# Initialize model
print("Loading Qwen3-VL-8B model...")
model = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    dtype="float16"
)
print("Model loaded!")

def handler(event):
    """
    Handler for Qwen3-VL vision-language understanding

    Input format:
    {
        "input": {
            "prompt": "What emotions are visible in this image?",
            "image": "base64_encoded_image",
            "max_tokens": 512,
            "temperature": 0.7
        }
    }
    """
    try:
        input_data = event.get('input', {})

        prompt_text = input_data.get('prompt', '')
        image_b64 = input_data.get('image')
        max_tokens = input_data.get('max_tokens', 512)
        temperature = input_data.get('temperature', 0.7)

        if not image_b64:
            return {"error": "No image provided"}

        # Decode image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        # Save temporarily (vLLM needs file path)
        temp_path = "/tmp/input_image.jpg"
        image.save(temp_path)

        # Create multimodal prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_path},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )

        # Generate
        outputs = model.generate(messages, sampling_params=sampling_params)

        response_text = outputs[0].outputs[0].text

        return {
            'response': response_text,
            'prompt_tokens': len(outputs[0].prompt_token_ids),
            'completion_tokens': len(outputs[0].outputs[0].token_ids)
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({'handler': handler})
```

### Estimated Costs
- **A100 40GB**: ~$0.003-0.005 per inference
- **A100 80GB**: ~$0.004-0.006 per inference

---

## Summary: Deployment Checklist

### Endpoint 1: YOLOv11
- [ ] Build Docker image with Ultralytics + YOLOv11m
- [ ] Push to Docker Hub
- [ ] Deploy on RunPod (RTX 4090 or A40)
- [ ] Test with sample image
- [ ] Save Endpoint ID to `.env`

### Endpoint 2: Faster-Whisper
- [ ] Use official RunPod template OR build custom
- [ ] Deploy on RunPod (A40 recommended)
- [ ] Test with sample audio
- [ ] Save Endpoint ID to `.env`

### Endpoint 3: Qwen3-VL-8B
- [ ] Deploy via vLLM Quick Deploy
- [ ] Configure environment variables
- [ ] Deploy on RunPod (A100 40GB minimum)
- [ ] Test with sample image + prompt
- [ ] Save Endpoint ID to `.env`

---

## Cost Estimates

### Per-Hour Processing (30fps video):
- **YOLOv11** (every 3rd frame = 10fps): ~$14.40/hour
- **Whisper** (continuous audio): ~$3.60/hour
- **Qwen3-VL** (every 30 frames = 1fps): ~$10.80/hour

**Total**: ~$29-35/hour of video processing

**Compare to**: Dedicated GPU server rental ($200-400/hour)

### Optimization Tips:
1. **Frame sampling**: Process fewer frames (e.g., 5fps instead of 10fps)
2. **Smart triggering**: Only call VLM when detections are interesting
3. **Batch processing**: Send multiple frames in one request where possible
4. **Auto-scaling**: Use min_workers=0 to scale to zero when idle

---

## Next Steps

After deploying all 3 endpoints:

1. Update `.env` file with all 3 endpoint IDs
2. Create client wrapper (`utils/runpod_client.py`)
3. Test each endpoint independently
4. Proceed to Phase 4: Build ingestion service
5. Phase 5-7: Build detection, audio, and VLM services (API clients)

---

## Resources

- RunPod Docs: https://docs.runpod.io/serverless
- vLLM Worker: https://github.com/runpod-workers/worker-vllm
- Faster-Whisper Worker: https://github.com/runpod-workers/worker-faster_whisper
- Ultralytics YOLOv11: https://docs.ultralytics.com/models/yolo11/
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL

---

**Ready to deploy?** Start with YOLOv11 endpoint first, test it thoroughly, then move to Whisper and finally Qwen3-VL.
