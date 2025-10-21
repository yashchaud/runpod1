# COMPLETE VIDEO UNDERSTANDING SYSTEM - BUILD GUIDE

## End-to-End Implementation Document

**OBJECTIVE**: Build a production-ready real-time video understanding agent that captures subtle emotions, understands silence context, and generates intelligent highlights from livestreams and video content.

---

## SYSTEM OVERVIEW

### Architecture Flow

```
WebRTC/RTMP Stream → LiveKit Gateway → Video/Audio Demux → Parallel Processing:
├─ Video Track → YOLOv11 Detection → Facial AU Detection → Micro-Expression Analysis
├─ Audio Track → Faster-Whisper ASR → Speaker Diarization → Prosody Analysis
└─ Fusion Layer → DeepAVFusion → Temporal Memory → VLM Reasoner → Event Store → Highlight Selector → Clip Export
```

### Technology Stack

- **Streaming**: LiveKit Cloud/Self-hosted + GStreamer
- **Detection**: YOLOv11m (Ultralytics), FG-Net, HTNet
- **Audio**: Faster-Whisper, pyannote-audio, librosa
- **VLM**: VideoChat-Flash-7B (vLLM deployment)
- **Storage**: PostgreSQL + TimescaleDB + S3
- **Orchestration**: Docker Compose
- **Language**: Python 3.11+

---

## PART 1: ENVIRONMENT SETUP

### 1.1 Hardware Requirements

**Minimum (Development)**:

- CPU: 16+ cores (Intel Xeon or AMD EPYC)
- RAM: 64GB
- GPU: NVIDIA RTX 4090 (24GB VRAM) or A6000
- Storage: 1TB NVMe SSD
- Network: 10Gbps

**Production**:

- GPU: NVIDIA A100 (40GB) or H100
- RAM: 128GB+
- Multi-GPU setup recommended

### 1.2 Base System Setup

```bash
# Ubuntu 22.04 LTS setup
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers and CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4 nvidia-driver-550

# Install Docker and NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install GStreamer
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-vaapi \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# Install system dependencies
sudo apt install -y \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopencv-dev \
    postgresql-14 \
    redis-server \
    nginx
```

### 1.3 Project Structure

```bash
mkdir -p ~/video-agent && cd ~/video-agent

# Create directory structure
mkdir -p {config,models,services,data,logs,outputs}
mkdir -p services/{ingestion,detection,audio,fusion,vlm,events,highlights,export}
mkdir -p data/{videos,frames,embeddings,events}
mkdir -p models/{yolo,emotion,whisper,vlm}

# Initialize git
git init
```

---

## PART 2: DATABASE & STORAGE SETUP

### 2.1 PostgreSQL + TimescaleDB

```bash
# Install TimescaleDB
sudo sh -c "echo 'deb [signed-by=/usr/share/keyrings/timescale.keyring] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/timescale.keyring
sudo apt update && sudo apt install -y timescaledb-2-postgresql-14
sudo timescaledb-tune --quiet --yes

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**Database Schema** (`config/schema.sql`):

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Streams table
CREATE TABLE streams (
    stream_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_name VARCHAR(255) NOT NULL,
    source_url TEXT,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Frames table (TimescaleDB hypertable)
CREATE TABLE frames (
    frame_id BIGSERIAL,
    stream_id UUID REFERENCES streams(stream_id),
    timestamp TIMESTAMPTZ NOT NULL,
    frame_number BIGINT NOT NULL,
    pts BIGINT,
    frame_data BYTEA,
    embeddings VECTOR(768),
    PRIMARY KEY (stream_id, timestamp, frame_id)
);
SELECT create_hypertable('frames', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Detections table
CREATE TABLE detections (
    detection_id BIGSERIAL,
    frame_id BIGINT,
    stream_id UUID REFERENCES streams(stream_id),
    timestamp TIMESTAMPTZ NOT NULL,
    object_type VARCHAR(100),
    bbox JSONB,
    confidence FLOAT,
    track_id INTEGER,
    features JSONB,
    PRIMARY KEY (stream_id, timestamp, detection_id)
);
SELECT create_hypertable('detections', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Emotions table
CREATE TABLE emotions (
    emotion_id BIGSERIAL,
    detection_id BIGINT,
    stream_id UUID REFERENCES streams(stream_id),
    timestamp TIMESTAMPTZ NOT NULL,
    person_track_id INTEGER,
    facial_aus JSONB,
    micro_expressions JSONB,
    emotion_labels JSONB,
    valence FLOAT,
    arousal FLOAT,
    PRIMARY KEY (stream_id, timestamp, emotion_id)
);
SELECT create_hypertable('emotions', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Audio segments table
CREATE TABLE audio_segments (
    segment_id BIGSERIAL,
    stream_id UUID REFERENCES streams(stream_id),
    timestamp TIMESTAMPTZ NOT NULL,
    start_time FLOAT,
    end_time FLOAT,
    speaker_id VARCHAR(50),
    text TEXT,
    confidence FLOAT,
    prosody JSONB,
    is_silence BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (stream_id, timestamp, segment_id)
);
SELECT create_hypertable('audio_segments', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Events table
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id UUID REFERENCES streams(stream_id),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    event_type VARCHAR(100),
    description TEXT,
    entities JSONB,
    scores JSONB,
    clip_worthy_score FLOAT,
    metadata JSONB
);
CREATE INDEX idx_events_stream_time ON events(stream_id, start_time);

-- Highlights table
CREATE TABLE highlights (
    highlight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id UUID REFERENCES streams(stream_id),
    event_id UUID REFERENCES events(event_id),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration INTERVAL,
    highlight_type VARCHAR(100),
    score FLOAT,
    exported BOOLEAN DEFAULT FALSE,
    export_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Memory store (for VLM context)
CREATE TABLE memory_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id UUID REFERENCES streams(stream_id),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    summary TEXT,
    token_compressed JSONB,
    embeddings VECTOR(4096)
);
```

### 2.2 Redis Setup

```bash
# Configure Redis for streaming data
sudo tee /etc/redis/redis.conf > /dev/null <<EOF
maxmemory 8gb
maxmemory-policy allkeys-lru
save ""
appendonly yes
appendfsync everysec
EOF

sudo systemctl restart redis
```

### 2.3 S3/MinIO for Video Storage

```yaml
# docker-compose.minio.yml
version: "3.8"
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: changeme123
    command: server /data --console-address ":9001"
    volumes:
      - ./data/minio:/data
```

---

## PART 3: MODEL DOWNLOADS & SETUP

### 3.1 Download All Required Models

```bash
# Create Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate optimum[onnxruntime-gpu]
pip install ultralytics opencv-python pillow
pip install huggingface-hub
```

**Model Download Script** (`scripts/download_models.py`):

```python
#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
from ultralytics import YOLO

MODELS_DIR = "./models"

def download_models():
    # YOLOv11
    print("Downloading YOLOv11m...")
    model = YOLO('yolo11m.pt')
    os.rename('yolo11m.pt', f'{MODELS_DIR}/yolo/yolo11m.pt')

    # YOLOv11-pose for body language
    model = YOLO('yolo11m-pose.pt')
    os.rename('yolo11m-pose.pt', f'{MODELS_DIR}/yolo/yolo11m-pose.pt')

    # Faster-Whisper models
    print("Downloading Whisper models...")
    os.system(f"ct2-transformers-converter --model openai/whisper-large-v3-turbo --output_dir {MODELS_DIR}/whisper/large-v3-turbo --quantization float16")

    # VideoChat-Flash
    print("Downloading VideoChat-Flash...")
    snapshot_download(
        repo_id="OpenGVLab/VideoChat-Flash-Qwen2_5-7B_res448",
        local_dir=f"{MODELS_DIR}/vlm/videochat-flash",
        local_dir_use_symlinks=False
    )

    # Qwen2.5-VL alternative
    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=f"{MODELS_DIR}/vlm/qwen2.5-vl-7b",
        local_dir_use_symlinks=False
    )

    # Emotion detection models (placeholder - you'll need to implement FG-Net/HTNet)
    # For production, use pre-trained AU detection models or Affectiva SDK
    print("Note: Implement custom emotion models or integrate Affectiva SDK")

    # Speaker diarization
    print("Downloading pyannote models...")
    snapshot_download(
        repo_id="pyannote/speaker-diarization-3.1",
        local_dir=f"{MODELS_DIR}/audio/pyannote",
        local_dir_use_symlinks=False
    )

    print("All models downloaded!")

if __name__ == "__main__":
    download_models()
```

```bash
python scripts/download_models.py
```

---

## PART 4: SERVICE IMPLEMENTATIONS

### 4.1 LiveKit Ingestion Service

**Install Dependencies**:

```bash
pip install livekit livekit-agents opencv-python av
```

**Service Code** (`services/ingestion/livekit_ingestion.py`):

```python
import asyncio
import logging
from livekit import rtc, api
import cv2
import numpy as np
import redis
import psycopg2
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitIngestion:
    def __init__(self, livekit_url, api_key, api_secret, redis_host='localhost'):
        self.url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        self.db = psycopg2.connect(
            dbname="videoagent",
            user="postgres",
            password="postgres",
            host="localhost"
        )

    async def start_ingestion(self, room_name, stream_id):
        """Connect to LiveKit room and start streaming frames"""
        room = rtc.Room()

        @room.on("track_subscribed")
        async def on_track_subscribed(track: rtc.Track, *_):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"Subscribed to video track: {track.sid}")
                asyncio.create_task(self._process_video_track(track, stream_id))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Subscribed to audio track: {track.sid}")
                asyncio.create_task(self._process_audio_track(track, stream_id))

        token = api.AccessToken(self.api_key, self.api_secret) \
            .with_identity("video-agent") \
            .with_grants(api.VideoGrants(room_join=True, room=room_name))

        await room.connect(self.url, token.to_jwt())
        logger.info(f"Connected to room: {room_name}")

        # Keep running
        while True:
            await asyncio.sleep(1)

    async def _process_video_track(self, track, stream_id):
        """Process incoming video frames"""
        video_stream = rtc.VideoStream(track)
        frame_count = 0

        async for frame in video_stream:
            try:
                # Convert to numpy array
                img = frame.convert(rtc.VideoBufferType.RGBA).data
                img_np = np.frombuffer(img, dtype=np.uint8).reshape(
                    (frame.height, frame.width, 4)
                )
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

                # Publish to Redis for downstream processing
                frame_data = {
                    'stream_id': stream_id,
                    'frame_number': frame_count,
                    'timestamp': datetime.utcnow().isoformat(),
                    'width': frame.width,
                    'height': frame.height,
                    'fps': 30  # Estimate or get from metadata
                }

                # Encode frame
                _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Push to Redis stream
                self.redis.xadd(
                    f'video:{stream_id}',
                    {
                        'metadata': json.dumps(frame_data),
                        'frame': buffer.tobytes()
                    },
                    maxlen=1000  # Keep last 1000 frames
                )

                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} video frames")

            except Exception as e:
                logger.error(f"Error processing video frame: {e}")

    async def _process_audio_track(self, track, stream_id):
        """Process incoming audio"""
        audio_stream = rtc.AudioStream(track)

        async for frame in audio_stream:
            try:
                # Convert to PCM bytes
                audio_data = frame.data.tobytes()

                # Push to Redis
                self.redis.xadd(
                    f'audio:{stream_id}',
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'sample_rate': frame.sample_rate,
                        'channels': frame.num_channels,
                        'samples': audio_data
                    },
                    maxlen=1000
                )

            except Exception as e:
                logger.error(f"Error processing audio frame: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python livekit_ingestion.py <url> <api_key> <api_secret> <room_name>")
        sys.exit(1)

    ingestion = LiveKitIngestion(sys.argv[1], sys.argv[2], sys.argv[3])

    # Create stream in DB
    import uuid
    stream_id = str(uuid.uuid4())

    asyncio.run(ingestion.start_ingestion(sys.argv[4], stream_id))
```

### 4.2 Detection Service (YOLOv11 + Tracking)

**Install Dependencies**:

```bash
pip install ultralytics supervision lap
```

**Service Code** (`services/detection/detector.py`):

```python
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import redis
import json
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionService:
    def __init__(self, model_path='./models/yolo/yolo11m.pt', redis_host='localhost'):
        self.model = YOLO(model_path)
        self.model.fuse()  # Optimize

        # ByteTrack tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        self.db = psycopg2.connect(
            dbname="videoagent",
            user="postgres",
            password="postgres",
            host="localhost"
        )

    def process_stream(self, stream_id):
        """Continuously process frames from Redis"""
        logger.info(f"Starting detection for stream: {stream_id}")

        last_id = '0'
        while True:
            try:
                # Read from Redis stream
                messages = self.redis.xread(
                    {f'video:{stream_id}': last_id},
                    count=1,
                    block=100
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        last_id = message_id

                        # Decode frame
                        metadata = json.loads(data[b'metadata'])
                        frame_bytes = data[b'frame']
                        frame = cv2.imdecode(
                            np.frombuffer(frame_bytes, np.uint8),
                            cv2.IMREAD_COLOR
                        )

                        # Run detection
                        detections = self._detect(frame, metadata)

                        # Store in DB
                        self._store_detections(stream_id, metadata, detections)

                        # Publish to Redis for downstream
                        self.redis.xadd(
                            f'detections:{stream_id}',
                            {
                                'metadata': json.dumps(metadata),
                                'detections': json.dumps(detections)
                            },
                            maxlen=1000
                        )

            except Exception as e:
                logger.error(f"Error in detection service: {e}")

    def _detect(self, frame, metadata):
        """Run YOLO detection and tracking"""
        results = self.model(frame, verbose=False)[0]

        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Update tracker
        detections = self.tracker.update_with_detections(detections)

        # Format output
        output = []
        for i, (bbox, conf, cls, track_id) in enumerate(zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            detections.tracker_id if detections.tracker_id is not None else [None]*len(detections)
        )):
            output.append({
                'bbox': bbox.tolist(),
                'confidence': float(conf),
                'class': int(cls),
                'class_name': self.model.names[int(cls)],
                'track_id': int(track_id) if track_id is not None else None
            })

        return output

    def _store_detections(self, stream_id, metadata, detections):
        """Store detections in PostgreSQL"""
        cursor = self.db.cursor()

        for det in detections:
            cursor.execute("""
                INSERT INTO detections
                (stream_id, timestamp, object_type, bbox, confidence, track_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                stream_id,
                metadata['timestamp'],
                det['class_name'],
                json.dumps(det['bbox']),
                det['confidence'],
                det['track_id']
            ))

        self.db.commit()
        cursor.close()

if __name__ == "__main__":
    import sys
    detector = DetectionService()
    detector.process_stream(sys.argv[1])
```

### 4.3 Audio Processing Service

**Install Dependencies**:

```bash
pip install faster-whisper pyannote.audio librosa soundfile
```

**Service Code** (`services/audio/audio_processor.py`):

```python
import numpy as np
from faster_whisper import WhisperModel
import librosa
import redis
import json
import psycopg2
from datetime import datetime
import logging
from io import BytesIO
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, model_path='./models/whisper/large-v3-turbo', redis_host='localhost'):
        self.whisper = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16"
        )

        # TODO: Initialize pyannote diarization
        # self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        self.db = psycopg2.connect(
            dbname="videoagent",
            user="postgres",
            password="postgres",
            host="localhost"
        )

        self.audio_buffer = []
        self.buffer_duration = 5.0  # Process every 5 seconds

    def process_stream(self, stream_id):
        """Process audio from Redis stream"""
        logger.info(f"Starting audio processing for stream: {stream_id}")

        last_id = '0'
        while True:
            try:
                messages = self.redis.xread(
                    {f'audio:{stream_id}': last_id},
                    count=10,
                    block=100
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        last_id = message_id

                        # Accumulate audio
                        audio_chunk = np.frombuffer(data[b'samples'], dtype=np.float32)
                        sample_rate = int(data[b'sample_rate'])

                        self.audio_buffer.append(audio_chunk)

                        # Check if we have enough audio
                        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                        duration = total_samples / sample_rate

                        if duration >= self.buffer_duration:
                            # Concatenate and process
                            audio = np.concatenate(self.audio_buffer)
                            self._process_audio_chunk(stream_id, audio, sample_rate)
                            self.audio_buffer = []

            except Exception as e:
                logger.error(f"Error in audio processing: {e}")

    def _process_audio_chunk(self, stream_id, audio, sample_rate):
        """Process accumulated audio chunk"""
        # Resample to 16kHz for Whisper
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Run Whisper transcription
        segments, info = self.whisper.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )

        cursor = self.db.cursor()

        for segment in segments:
            # Analyze prosody
            prosody = self._analyze_prosody(
                audio[int(segment.start*sample_rate):int(segment.end*sample_rate)],
                sample_rate
            )

            # Detect silence/pauses
            is_silence = segment.text.strip() == "" or len(segment.text.strip()) < 3

            # Store
            cursor.execute("""
                INSERT INTO audio_segments
                (stream_id, timestamp, start_time, end_time, text, confidence, prosody, is_silence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                stream_id,
                datetime.utcnow(),
                segment.start,
                segment.end,
                segment.text,
                segment.avg_logprob,
                json.dumps(prosody),
                is_silence
            ))

            logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        self.db.commit()
        cursor.close()

    def _analyze_prosody(self, audio, sr):
        """Extract prosodic features"""
        # F0 (pitch)
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
        f0_mean = float(np.mean(f0[f0 > 0])) if len(f0[f0 > 0]) > 0 else 0
        f0_std = float(np.std(f0[f0 > 0])) if len(f0[f0 > 0]) > 0 else 0

        # Energy
        energy = librosa.feature.rms(y=audio)[0]
        energy_mean = float(np.mean(energy))

        # Speaking rate (approximate via zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        return {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'energy_mean': energy_mean,
            'zcr_mean': float(np.mean(zcr))
        }

if __name__ == "__main__":
    import sys
    processor = AudioProcessor()
    processor.process_stream(sys.argv[1])
```

### 4.4 VLM Reasoning Service

**Install Dependencies**:

```bash
pip install vllm transformers accelerate
```

**Service Code** (`services/vlm/vlm_reasoner.py`):

```python
from vllm import LLM, SamplingParams
import redis
import json
import psycopg2
from datetime import datetime
import logging
import base64
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMReasoner:
    def __init__(self, model_path='./models/vlm/videochat-flash', redis_host='localhost'):
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.8,
            max_model_len=32768,
            dtype="float16",
            tensor_parallel_size=1  # Increase for multi-GPU
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        self.db = psycopg2.connect(
            dbname="videoagent",
            user="postgres",
            password="postgres",
            host="localhost"
        )

        self.context_window = []
        self.max_context_frames = 100

    def process_stream(self, stream_id):
        """Continuously reason about video stream"""
        logger.info(f"Starting VLM reasoning for stream: {stream_id}")

        last_video_id = '0'
        last_detection_id = '0'

        frame_count = 0

        while True:
            try:
                # Read frames
                video_msgs = self.redis.xread(
                    {f'video:{stream_id}': last_video_id},
                    count=1,
                    block=100
                )

                # Read detections
                detection_msgs = self.redis.xread(
                    {f'detections:{stream_id}': last_detection_id},
                    count=1,
                    block=100
                )

                if video_msgs:
                    for _, messages in video_msgs:
                        for msg_id, data in messages:
                            last_video_id = msg_id

                            # Sample frames (every 3rd frame for VLM = ~10fps at 30fps input)
                            frame_count += 1
                            if frame_count % 3 != 0:
                                continue

                            metadata = json.loads(data[b'metadata'])
                            frame_bytes = data[b'frame']

                            # Add to context
                            self.context_window.append({
                                'timestamp': metadata['timestamp'],
                                'frame': frame_bytes
                            })

                            # Keep window size
                            if len(self.context_window) > self.max_context_frames:
                                self.context_window.pop(0)

                            # Generate description every 30 frames (~1 second)
                            if frame_count % 30 == 0:
                                description = self._reason_about_context(stream_id)
                                self._store_event(stream_id, description)

            except Exception as e:
                logger.error(f"Error in VLM reasoning: {e}")

    def _reason_about_context(self, stream_id):
        """Generate description of current context"""
        # For VideoChat-Flash, we need to format the prompt appropriately
        # This is a simplified version - actual implementation depends on model

        prompt = """Analyze this video segment and describe:
1. What actions are happening?
2. What emotions are people displaying?
3. Is there anything noteworthy or unusual?
4. What is the overall mood/atmosphere?

Be concise but capture important details."""

        # TODO: Implement proper multi-frame encoding for VideoChat-Flash
        # For now, use last frame as placeholder

        try:
            outputs = self.llm.generate(
                [prompt],
                self.sampling_params
            )

            description = outputs[0].outputs[0].text
            logger.info(f"VLM Description: {description}")

            return description

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return ""

    def _store_event(self, stream_id, description):
        """Store event in database"""
        cursor = self.db.cursor()

        cursor.execute("""
            INSERT INTO events
            (stream_id, start_time, event_type, description)
            VALUES (%s, %s, %s, %s)
        """, (
            stream_id,
            datetime.utcnow(),
            'vlm_description',
            description
        ))

        self.db.commit()
        cursor.close()

if __name__ == "__main__":
    import sys
    reasoner = VLMReasoner()
    reasoner.process_stream(sys.argv[1])
```

---

## PART 5: ORCHESTRATION & DEPLOYMENT

### 5.1 Docker Compose Configuration

**Complete Stack** (`docker-compose.yml`):

```yaml
version: "3.8"

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: videoagent
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./config/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 8gb --maxmemory-policy allkeys-lru
    volumes:
      - ./data/redis:/data
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: changeme123
    command: server /data --console-address ":9001"
    volumes:
      - ./data/minio:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  ingestion:
    build:
      context: .
      dockerfile: services/ingestion/Dockerfile
    runtime: nvidia
    environment:
      LIVEKIT_URL: ${LIVEKIT_URL}
      LIVEKIT_API_KEY: ${LIVEKIT_API_KEY}
      LIVEKIT_API_SECRET: ${LIVEKIT_API_SECRET}
      REDIS_HOST: redis
    depends_on:
      - redis
      - postgres

  detection:
    build:
      context: .
      dockerfile: services/detection/Dockerfile
    runtime: nvidia
    environment:
      REDIS_HOST: redis
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  audio:
    build:
      context: .
      dockerfile: services/audio/Dockerfile
    runtime: nvidia
    environment:
      REDIS_HOST: redis
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
      - postgres

  vlm:
    build:
      context: .
      dockerfile: services/vlm/Dockerfile
    runtime: nvidia
    environment:
      REDIS_HOST: redis
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 5.2 Dockerfiles

**Base Dockerfile** (`Dockerfile.base`):

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["bash"]
```

**Requirements File** (`requirements.txt`):

```txt
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
transformers==4.44.0
accelerate==0.33.0
optimum[onnxruntime-gpu]==1.21.0
ultralytics==8.2.90
opencv-python==4.10.0.84
pillow==10.4.0
livekit==0.17.0
livekit-agents==0.9.0
faster-whisper==1.0.3
pyannote.audio==3.3.1
librosa==0.10.2
soundfile==0.12.1
vllm==0.5.5
supervision==0.22.0
redis==5.0.8
psycopg2-binary==2.9.9
numpy==1.26.4
scipy==1.14.1
scikit-learn==1.5.1
pandas==2.2.2
```

### 5.3 Startup Scripts

**Main Launcher** (`start.sh`):

```bash
#!/bin/bash

set -e

echo "Starting Video Understanding Agent..."

# Start infrastructure
docker-compose up -d postgres redis minio

# Wait for services
sleep 10

# Initialize database
psql -h localhost -U postgres -d videoagent -f config/schema.sql

# Start processing services
docker-compose up -d detection audio vlm

echo "System ready!"
echo "Connect LiveKit stream to start processing"
```

---

## PART 6: TESTING & VALIDATION

### 6.1 End-to-End Test

**Test Script** (`tests/e2e_test.py`):

```python
import cv2
import redis
import psycopg2
import time
import json
import uuid

def test_pipeline():
    # Create test stream
    stream_id = str(uuid.uuid4())

    # Connect to services
    r = redis.Redis(decode_responses=False)
    db = psycopg2.connect(dbname="videoagent", user="postgres", password="postgres")
    cursor = db.cursor()

    # Register stream
    cursor.execute("""
        INSERT INTO streams (stream_id, stream_name, source_url)
        VALUES (%s, %s, %s)
    """, (stream_id, 'test_stream', 'file://test.mp4'))
    db.commit()

    # Load test video
    cap = cv2.VideoCapture('test_video.mp4')
    frame_count = 0

    print("Injecting test video frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)

        # Push to Redis
        r.xadd(
            f'video:{stream_id}',
            {
                'metadata': json.dumps({
                    'stream_id': stream_id,
                    'frame_number': frame_count,
                    'timestamp': time.time(),
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                }),
                'frame': buffer.tobytes()
            }
        )

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Injected {frame_count} frames")

        time.sleep(0.033)  # ~30fps

    cap.release()

    # Wait for processing
    print("Waiting for processing...")
    time.sleep(30)

    # Verify results
    cursor.execute("SELECT COUNT(*) FROM detections WHERE stream_id = %s", (stream_id,))
    detection_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audio_segments WHERE stream_id = %s", (stream_id,))
    audio_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM events WHERE stream_id = %s", (stream_id,))
    event_count = cursor.fetchone()[0]

    print(f"\n=== Test Results ===")
    print(f"Frames injected: {frame_count}")
    print(f"Detections found: {detection_count}")
    print(f"Audio segments: {audio_count}")
    print(f"Events generated: {event_count}")

    cursor.close()
    db.close()

if __name__ == "__main__":
    test_pipeline()
```

---

## PART 7: MONITORING & LOGGING

### 7.1 Prometheus Metrics

**Add to each service**:

```python
from prometheus_client import Counter, Histogram, start_http_server

FRAMES_PROCESSED = Counter('frames_processed_total', 'Total frames processed')
PROCESSING_TIME = Histogram('processing_seconds', 'Frame processing time')

# Start metrics server
start_http_server(8000)

# In processing loop:
with PROCESSING_TIME.time():
    # Process frame
    pass
FRAMES_PROCESSED.inc()
```

### 7.2 Grafana Dashboard

**Save as** `config/grafana-dashboard.json` (basic template):

```json
{
  "dashboard": {
    "title": "Video Agent Pipeline",
    "panels": [
      {
        "title": "Frames Processed/sec",
        "targets": [{ "expr": "rate(frames_processed_total[1m])" }]
      },
      {
        "title": "Processing Latency",
        "targets": [{ "expr": "processing_seconds" }]
      }
    ]
  }
}
```

---

## PART 8: PRODUCTION DEPLOYMENT CHECKLIST

### 8.1 Pre-Deployment

- [ ] Download all models
- [ ] Configure environment variables
- [ ] Set up PostgreSQL with schema
- [ ] Configure Redis persistence
- [ ] Set up S3/MinIO buckets
- [ ] Test with sample video
- [ ] Configure monitoring
- [ ] Set up log aggregation

### 8.2 Security

```bash
# Generate secure passwords
openssl rand -base64 32  # For DB password
openssl rand -base64 32  # For MinIO

# Use secrets management
# Store in .env file (DO NOT COMMIT)
```

### 8.3 Scaling

**Horizontal Scaling**:

```yaml
# In docker-compose.yml, add replicas:
detection:
  deploy:
    replicas: 3
```

**GPU Allocation**:

- Detection: 1 GPU
- Audio: 0.5 GPU (shared)
- VLM: 1-2 GPUs (high memory)

---

## PART 9: API LAYER (BONUS)

### 9.1 FastAPI Service

**API Service** (`services/api/main.py`):

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import psycopg2
import json

app = FastAPI()

@app.get("/streams")
def list_streams():
    db = psycopg2.connect(dbname="videoagent", user="postgres", password="postgres")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM streams")
    streams = cursor.fetchall()
    cursor.close()
    return {"streams": streams}

@app.get("/events/{stream_id}")
def get_events(stream_id: str, limit: int = 100):
    db = psycopg2.connect(dbname="videoagent", user="postgres", password="postgres")
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM events
        WHERE stream_id = %s
        ORDER BY start_time DESC
        LIMIT %s
    """, (stream_id, limit))
    events = cursor.fetchall()
    cursor.close()
    return {"events": events}

@app.get("/highlights/{stream_id}")
def get_highlights(stream_id: str):
    db = psycopg2.connect(dbname="videoagent", user="postgres", password="postgres")
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM highlights
        WHERE stream_id = %s
        ORDER BY score DESC
    """, (stream_id,))
    highlights = cursor.fetchall()
    cursor.close()
    return {"highlights": highlights}

@app.websocket("/ws/events/{stream_id}")
async def websocket_events(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    # Stream real-time events
    # TODO: Implement Redis pub/sub listener
```

---

## COMPLETE EXECUTION COMMANDS

```bash
# 1. Clone and setup
git clone <your-repo> && cd video-agent
python3.11 -m venv venv && source venv/bin/activate

# 2. Download models
python scripts/download_models.py

# 3. Start infrastructure
docker-compose up -d postgres redis minio

# 4. Initialize database
psql -h localhost -U postgres -d videoagent -f config/schema.sql

# 5. Start services
docker-compose up -d

# 6. Run test
python tests/e2e_test.py

# 7. Monitor logs
docker-compose logs -f
```

---

## NEXT STEPS FOR COMPLETION

**Critical Missing Pieces to Implement**:

1. **Emotion Detection Models**:

   - Integrate FG-Net or use Affectiva SDK
   - Implement HTNet for micro-expressions
   - Alternative: Use pre-trained FER models from HuggingFace

2. **DeepAVFusion Integration**:

   - Clone repo: https://github.com/stoneMo/DeepAVFusion
   - Adapt for streaming inference
   - Integrate with audio/video pipelines

3. **Highlight Selector Logic**:

   - Implement scoring algorithm
   - Add clip cutting with ffmpeg
   - Export to S3

4. **Production VLM Setup**:

   - Properly configure VideoChat-Flash
   - Implement frame batching
   - Add compression strategies

5. **Speaker Diarization**:

   - Integrate pyannote.audio
   - Link with visual face tracking

6. **Real LiveKit Integration**:
   - Get LiveKit Cloud account or deploy self-hosted
   - Configure webhooks
   - Handle reconnections

**THIS DOCUMENT PROVIDES**:
✅ Complete infrastructure setup
✅ Database schema
✅ Core services skeleton
✅ Docker orchestration
✅ Testing framework
✅ Deployment scripts

**YOU NEED TO ADD**:
❌ Actual emotion model implementations
❌ Full VLM prompt engineering
❌ Highlight scoring algorithm
❌ Video export pipeline
❌ Production credentials

Give this document to Claude and ask: "Implement the missing pieces marked with ❌, starting with emotion detection integration"
