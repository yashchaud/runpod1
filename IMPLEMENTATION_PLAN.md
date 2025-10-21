# VIDEO UNDERSTANDING AGENT - IMPLEMENTATION TASK BREAKDOWN

## ARCHITECTURE: RunPod Serverless Endpoints

**Key Change**: All ML models (YOLO, Whisper, VLM) will be deployed as **RunPod Serverless Endpoints** instead of local GPU hosting. This means:
- ✅ No need for expensive local GPUs (RTX 4090, A100)
- ✅ Auto-scaling based on demand
- ✅ Pay-per-second pricing
- ✅ Services call RunPod APIs via HTTP
- ✅ Simplified deployment (no Docker GPU orchestration)

**Purpose**: This document breaks down the PRD into granular, testable tasks that can be completed and verified one at a time. Each task includes a test/verification step to ensure nothing is built on a broken foundation.

**How to Use This Document**:
- Work through phases sequentially
- Check off tasks as you complete them
- Run verification steps before moving to the next task
- If a test fails, fix it before proceeding

---

## PHASE 0: PRE-FLIGHT CHECKS

### RunPod Account Setup
- [ ] **Task 0.1**: Create RunPod account
  - **Action**: Sign up at https://www.runpod.io/
  - **Test**: Login to RunPod dashboard
  - **Expected**: Account created successfully

- [ ] **Task 0.2**: Get RunPod API Key
  - **Action**: Navigate to Settings → API Keys in RunPod dashboard
  - **Test**: Copy API key to safe location
  - **Expected**: API key available

- [ ] **Task 0.3**: Add billing/credits to RunPod
  - **Action**: Add payment method or credits
  - **Test**: Check account balance
  - **Expected**: Credits available for serverless usage

### Local Environment Verification
- [ ] **Task 0.4**: Check available disk space (for data/logs only, not models)
  - **Test**: `df -h` shows at least 500GB free space
  - **Expected**: Sufficient space for video data and database

- [ ] **Task 0.5**: Verify network connectivity
  - **Test**: `ping -c 3 8.8.8.8` and `curl -I https://api.runpod.ai`
  - **Expected**: Successful responses to internet and RunPod API

---

## PHASE 1: ENVIRONMENT SETUP

### 1.1 Base System Setup (CPU-Only, No GPU Required)
- [ ] **Task 1.1.1**: Update system packages
  - **Command**: `sudo apt update && sudo apt upgrade -y` (Linux) or `brew update` (Mac)
  - **Test**: `apt list --upgradable` shows no pending updates
  - **Expected**: System is up to date

- [ ] **Task 1.1.2**: Install Docker (for local services only)
  - **Command**: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`
  - **Test**: `docker --version` and `docker run hello-world`
  - **Expected**: Docker runs successfully
  - **Note**: No GPU support needed - models run on RunPod

- [ ] **Task 1.1.3**: Install Python 3.11
  - **Command**: `sudo apt install -y python3.11 python3.11-venv python3.11-dev`
  - **Test**: `python3.11 --version`
  - **Expected**: Python 3.11.x installed

- [ ] **Task 1.1.4**: Install GStreamer (for video processing)
  - **Command**: `sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav`
  - **Test**: `gst-launch-1.0 --version`
  - **Expected**: GStreamer 1.x installed

- [ ] **Task 1.1.5**: Install system dependencies
  - **Command**: `sudo apt install -y build-essential cmake git ffmpeg libsm6 libxext6`
  - **Test**: `ffmpeg -version`, `cmake --version`, `git --version`
  - **Expected**: All tools available

### 1.2 Project Structure
- [ ] **Task 1.2.1**: Create project directory structure
  - **Command**: Run directory creation commands from PRD
  - **Test**: `tree -L 2 ~/video-agent` (or `ls -R`)
  - **Expected**: All required directories exist

- [ ] **Task 1.2.2**: Initialize git repository
  - **Command**: `git init` in project directory
  - **Test**: `git status` works
  - **Expected**: Git repository initialized

- [ ] **Task 1.2.3**: Create .gitignore
  - **Content**: Add `venv/`, `*.pyc`, `__pycache__/`, `data/`, `models/`, `.env`
  - **Test**: `git status` doesn't show ignored files
  - **Expected**: Proper files ignored

### 1.3 Python Environment
- [ ] **Task 1.3.1**: Create Python virtual environment
  - **Command**: `python3.11 -m venv venv`
  - **Test**: `source venv/bin/activate && which python`
  - **Expected**: Python points to venv

- [ ] **Task 1.3.2**: Upgrade pip, wheel, setuptools
  - **Command**: `pip install --upgrade pip wheel setuptools`
  - **Test**: `pip --version`
  - **Expected**: Latest pip version

- [ ] **Task 1.3.3**: Install RunPod Python SDK
  - **Command**: `pip install runpod requests`
  - **Test**: `python -c "import runpod; print('OK')"`
  - **Expected**: RunPod SDK imports successfully

- [ ] **Task 1.3.4**: Install basic ML/CV libraries (CPU versions only)
  - **Command**: `pip install opencv-python pillow numpy`
  - **Test**: `python -c "import cv2; import PIL; import numpy; print('OK')"`
  - **Expected**: Libraries import successfully
  - **Note**: No PyTorch/CUDA needed locally - inference happens on RunPod

---

## PHASE 2: DATABASE & STORAGE SETUP

### 2.1 PostgreSQL + TimescaleDB
- [ ] **Task 2.1.1**: Install PostgreSQL 14
  - **Command**: `sudo apt install -y postgresql-14`
  - **Test**: `sudo systemctl status postgresql`
  - **Expected**: PostgreSQL running

- [ ] **Task 2.1.2**: Install TimescaleDB extension
  - **Command**: Follow TimescaleDB installation from PRD
  - **Test**: `sudo -u postgres psql -c "SELECT default_version FROM pg_available_extensions WHERE name='timescaledb';"`
  - **Expected**: TimescaleDB version shown

- [ ] **Task 2.1.3**: Configure PostgreSQL for TimescaleDB
  - **Command**: `sudo timescaledb-tune --quiet --yes`
  - **Test**: Check `/etc/postgresql/14/main/postgresql.conf` for timescaledb in shared_preload_libraries
  - **Expected**: TimescaleDB configured

- [ ] **Task 2.1.4**: Restart PostgreSQL
  - **Command**: `sudo systemctl restart postgresql`
  - **Test**: `sudo systemctl status postgresql`
  - **Expected**: Service running without errors

- [ ] **Task 2.1.5**: Create database
  - **Command**: `sudo -u postgres createdb videoagent`
  - **Test**: `sudo -u postgres psql -l | grep videoagent`
  - **Expected**: Database exists

- [ ] **Task 2.1.6**: Create schema.sql file
  - **File**: `config/schema.sql` with content from PRD
  - **Test**: File exists and is valid SQL
  - **Expected**: File created

- [ ] **Task 2.1.7**: Apply database schema
  - **Command**: `sudo -u postgres psql -d videoagent -f config/schema.sql`
  - **Test**: `sudo -u postgres psql -d videoagent -c "\dt"`
  - **Expected**: All tables created (streams, frames, detections, emotions, audio_segments, events, highlights, memory_chunks)

- [ ] **Task 2.1.8**: Verify hypertables
  - **Command**: `sudo -u postgres psql -d videoagent -c "SELECT * FROM timescaledb_information.hypertables;"`
  - **Test**: Query runs successfully
  - **Expected**: frames, detections, emotions, audio_segments are hypertables

### 2.2 Redis Setup
- [ ] **Task 2.2.1**: Install Redis
  - **Command**: `sudo apt install -y redis-server`
  - **Test**: `redis-cli ping`
  - **Expected**: Returns `PONG`

- [ ] **Task 2.2.2**: Configure Redis
  - **File**: Update `/etc/redis/redis.conf` per PRD
  - **Test**: `sudo systemctl restart redis && redis-cli INFO | grep maxmemory`
  - **Expected**: Shows 8GB limit

- [ ] **Task 2.2.3**: Test Redis persistence
  - **Command**: `redis-cli SET test "hello" && redis-cli GET test`
  - **Test**: Returns "hello"
  - **Expected**: Redis working

### 2.3 MinIO (S3 Storage)
- [ ] **Task 2.3.1**: Create docker-compose.minio.yml
  - **File**: Create file from PRD
  - **Test**: File exists and is valid YAML
  - **Expected**: Valid docker-compose file

- [ ] **Task 2.3.2**: Start MinIO container
  - **Command**: `docker-compose -f docker-compose.minio.yml up -d`
  - **Test**: `docker ps | grep minio`
  - **Expected**: Container running

- [ ] **Task 2.3.3**: Test MinIO access
  - **Browser**: Open http://localhost:9001
  - **Test**: Login with admin/changeme123
  - **Expected**: MinIO console accessible

- [ ] **Task 2.3.4**: Create buckets
  - **Command**: Create buckets: `videos`, `frames`, `highlights`
  - **Test**: Buckets visible in console
  - **Expected**: All buckets created

---

## PHASE 3: RUNPOD SERVERLESS ENDPOINT SETUP

### 3.1 YOLOv11 Detection Endpoint
- [ ] **Task 3.1.1**: Deploy YOLOv11 serverless endpoint on RunPod
  - **Action**: In RunPod dashboard, go to Serverless → Deploy
  - **Template**: Search for "YOLOv11" or use custom template
  - **GPU**: Select GPU type (e.g., RTX 4090 or A40)
  - **Test**: Endpoint status shows "Running"
  - **Expected**: Endpoint URL available

- [ ] **Task 3.1.2**: Get YOLOv11 endpoint credentials
  - **Action**: Copy Endpoint ID and API key
  - **Test**: Save to `.env` file as `RUNPOD_YOLO_ENDPOINT_ID` and `RUNPOD_API_KEY`
  - **Expected**: Credentials stored securely

- [ ] **Task 3.1.3**: Test YOLOv11 endpoint
  - **Script**: Create `scripts/test_yolo_endpoint.py`
  - **Test**: Send test image to endpoint, verify response
  - **Expected**: Returns bounding boxes in JSON format

### 3.2 Whisper ASR Endpoint
- [ ] **Task 3.2.1**: Deploy Whisper Large V3 Turbo endpoint
  - **Action**: Deploy serverless endpoint for Whisper
  - **Template**: Use "Faster Whisper" or custom Whisper template
  - **GPU**: Select GPU (A40 or A100 recommended for faster inference)
  - **Test**: Endpoint status shows "Running"
  - **Expected**: Endpoint URL available

- [ ] **Task 3.2.2**: Get Whisper endpoint credentials
  - **Action**: Copy Endpoint ID
  - **Test**: Save to `.env` as `RUNPOD_WHISPER_ENDPOINT_ID`
  - **Expected**: Credentials stored

- [ ] **Task 3.2.3**: Test Whisper endpoint
  - **Script**: Create `scripts/test_whisper_endpoint.py`
  - **Test**: Send test audio file, verify transcription
  - **Expected**: Returns accurate transcription with timestamps

### 3.3 VLM (Vision Language Model) Endpoint
- [ ] **Task 3.3.1**: Deploy Qwen2.5-VL or VideoChat-Flash endpoint
  - **Action**: Deploy VLM serverless endpoint
  - **Template**: Use Qwen2.5-VL-7B or custom VLM template
  - **GPU**: Select high-memory GPU (A100 40GB recommended)
  - **Test**: Endpoint status shows "Running"
  - **Expected**: Endpoint URL available

- [ ] **Task 3.3.2**: Get VLM endpoint credentials
  - **Action**: Copy Endpoint ID
  - **Test**: Save to `.env` as `RUNPOD_VLM_ENDPOINT_ID`
  - **Expected**: Credentials stored

- [ ] **Task 3.3.3**: Test VLM endpoint
  - **Script**: Create `scripts/test_vlm_endpoint.py`
  - **Test**: Send test image with prompt, verify description
  - **Expected**: Returns coherent video/image description

### 3.4 Optional: Emotion Detection Endpoint
- [ ] **Task 3.4.1**: Deploy FER (Facial Expression Recognition) endpoint
  - **Action**: Deploy emotion detection endpoint (optional)
  - **Template**: Use custom FER model or pre-built template
  - **Test**: Endpoint status shows "Running"
  - **Expected**: Endpoint URL available

- [ ] **Task 3.4.2**: Test emotion endpoint
  - **Script**: Test with face images
  - **Expected**: Returns emotion labels (happy, sad, angry, etc.)

### 3.5 Endpoint Configuration File
- [ ] **Task 3.5.1**: Create config/runpod_endpoints.json
  - **File**: Store all endpoint configurations
  - **Content**: JSON with endpoint IDs, models, GPU types
  - **Test**: File is valid JSON
  - **Expected**: Central configuration for all endpoints

- [ ] **Task 3.5.2**: Create RunPod client wrapper
  - **File**: `utils/runpod_client.py`
  - **Code**: Wrapper class for making RunPod API calls
  - **Test**: `python -c "from utils.runpod_client import RunPodClient; print('OK')"`
  - **Expected**: Reusable client for all services

---

## PHASE 4: INGESTION SERVICE

### 4.1 Service Skeleton
- [ ] **Task 4.1.1**: Install LiveKit dependencies
  - **Command**: `pip install livekit livekit-agents opencv-python av redis psycopg2-binary`
  - **Test**: `python -c "import livekit; print('OK')"`
  - **Expected**: Packages installed

- [ ] **Task 4.1.2**: Create services/ingestion/livekit_ingestion.py
  - **File**: Create skeleton class from PRD
  - **Test**: `python services/ingestion/livekit_ingestion.py --help` shows usage
  - **Expected**: No syntax errors

- [ ] **Task 4.1.3**: Implement database connection
  - **Code**: Add psycopg2 connection in __init__
  - **Test**: Run with test DB credentials
  - **Expected**: Connects successfully

- [ ] **Task 4.1.4**: Implement Redis connection
  - **Code**: Add Redis connection in __init__
  - **Test**: Test writing to Redis stream
  - **Expected**: Can write to Redis

### 4.2 Video Track Processing
- [ ] **Task 4.2.1**: Implement _process_video_track method
  - **Code**: Add video frame processing logic
  - **Test**: Use test video file
  - **Expected**: Frames decoded to numpy arrays

- [ ] **Task 4.2.2**: Implement frame encoding to JPEG
  - **Code**: Add cv2.imencode logic
  - **Test**: Verify encoded frame is valid JPEG
  - **Expected**: Frame compresses correctly

- [ ] **Task 4.2.3**: Implement Redis publishing
  - **Code**: Add xadd to push frames
  - **Test**: Check Redis stream has entries: `redis-cli XLEN video:test-stream`
  - **Expected**: Frame count increases

### 4.3 Audio Track Processing
- [ ] **Task 4.3.1**: Implement _process_audio_track method
  - **Code**: Add audio processing logic
  - **Test**: Process test audio file
  - **Expected**: Audio frames published to Redis

- [ ] **Task 4.3.2**: Test audio format conversion
  - **Code**: Ensure PCM format is correct
  - **Test**: Decode audio from Redis and verify it's playable
  - **Expected**: Audio data is valid

### 4.4 Integration Test
- [ ] **Task 4.4.1**: Create mock LiveKit test
  - **Script**: Create test that simulates LiveKit room
  - **Test**: Run ingestion service with mock data
  - **Expected**: Data flows to Redis

- [ ] **Task 4.4.2**: Test with real video file (fallback mode)
  - **Script**: Add file input mode for testing
  - **Test**: Process sample MP4 file
  - **Expected**: Frames and audio extracted successfully

---

## PHASE 5: DETECTION SERVICE (RunPod API Client)

### 5.1 Service Skeleton
- [ ] **Task 5.1.1**: Install detection dependencies
  - **Command**: `pip install supervision opencv-python requests runpod`
  - **Test**: `python -c "import supervision; print('OK')"`
  - **Expected**: Packages installed
  - **Note**: No Ultralytics/YOLO locally - detection happens on RunPod

- [ ] **Task 5.1.2**: Create services/detection/detector.py
  - **File**: Create skeleton that calls RunPod YOLO endpoint
  - **Test**: Script runs without errors
  - **Expected**: No syntax errors

- [ ] **Task 5.1.3**: Initialize RunPod client
  - **Code**: Initialize RunPodClient for YOLO endpoint
  - **Test**: Test connection to RunPod endpoint
  - **Expected**: Successful connection, no auth errors

### 5.2 Detection Logic (API Calls)
- [ ] **Task 5.2.1**: Implement _detect method (RunPod API call)
  - **Code**: Encode frame to base64, send to RunPod YOLO endpoint
  - **Test**: Send test image to RunPod, parse response
  - **Expected**: Returns bounding boxes from RunPod API

- [ ] **Task 5.2.2**: Implement local ByteTrack tracker
  - **Code**: Add tracker initialization (tracking happens locally)
  - **Test**: Track objects across multiple frames from RunPod detections
  - **Expected**: Consistent track IDs across frames
  - **Note**: Detection is remote, tracking is local for continuity

- [ ] **Task 5.2.3**: Test tracking on video
  - **Test**: Process 100-frame sequence via RunPod
  - **Expected**: Track IDs persist across frames, API latency < 200ms

### 5.3 Redis Integration
- [ ] **Task 5.3.1**: Implement Redis stream reading
  - **Code**: Add XREAD logic in process_stream
  - **Test**: Read frames from ingestion service output
  - **Expected**: Consumes frames successfully

- [ ] **Task 5.3.2**: Implement detection publishing
  - **Code**: Publish detections to `detections:{stream_id}` stream
  - **Test**: `redis-cli XREAD STREAMS detections:test-stream 0`
  - **Expected**: Detections visible in Redis

### 5.4 Database Storage
- [ ] **Task 5.4.1**: Implement _store_detections method
  - **Code**: Add PostgreSQL insert logic
  - **Test**: Query database after processing: `SELECT COUNT(*) FROM detections`
  - **Expected**: Detections stored in DB

- [ ] **Task 5.4.2**: Test batch insert performance
  - **Test**: Process 1000 frames, measure DB write time
  - **Expected**: < 100ms per batch

### 5.5 End-to-End Test
- [ ] **Task 5.5.1**: Run detection service standalone
  - **Command**: `python services/detection/detector.py test-stream-id`
  - **Test**: Process frames from Redis, store in DB
  - **Expected**: Service runs continuously without errors

- [ ] **Task 5.5.2**: Verify detection accuracy
  - **Test**: Process known test video with people
  - **Expected**: Detects people with >70% confidence

---

## PHASE 6: AUDIO PROCESSING SERVICE (RunPod API Client)

### 6.1 Service Skeleton
- [ ] **Task 6.1.1**: Install audio dependencies
  - **Command**: `pip install librosa soundfile requests runpod`
  - **Test**: `python -c "import librosa; print('OK')"`
  - **Expected**: Packages installed
  - **Note**: No Whisper locally - transcription happens on RunPod

- [ ] **Task 6.1.2**: Create services/audio/audio_processor.py
  - **File**: Create service that calls RunPod Whisper endpoint
  - **Test**: Script imports successfully
  - **Expected**: No syntax errors

- [ ] **Task 6.1.3**: Initialize RunPod Whisper client
  - **Code**: Initialize RunPodClient for Whisper endpoint
  - **Test**: Test connection to endpoint
  - **Expected**: Successful connection

### 6.2 Transcription Logic (API Calls)
- [ ] **Task 6.2.1**: Implement audio buffering
  - **Code**: Add buffer accumulation in process_stream
  - **Test**: Accumulate 5 seconds of audio
  - **Expected**: Buffer fills correctly

- [ ] **Task 6.2.2**: Implement _process_audio_chunk method (RunPod API)
  - **Code**: Encode audio to base64, send to RunPod Whisper endpoint
  - **Test**: Send test audio file to RunPod, parse response
  - **Expected**: Returns accurate transcription with timestamps from RunPod

- [ ] **Task 6.2.3**: Test VAD (Voice Activity Detection) via RunPod
  - **Test**: Process audio with silence
  - **Expected**: RunPod Whisper returns only speech segments (VAD built-in)
  - **Note**: VAD happens on RunPod side

### 6.3 Prosody Analysis
- [ ] **Task 6.3.1**: Implement _analyze_prosody method
  - **Code**: Add F0, energy, ZCR extraction
  - **Test**: Analyze speech sample
  - **Expected**: Returns valid prosody features

- [ ] **Task 6.3.2**: Test silence detection
  - **Test**: Process silent audio segment
  - **Expected**: is_silence flag set correctly

### 6.4 Database Storage
- [ ] **Task 6.4.1**: Implement database inserts
  - **Code**: Store audio_segments in DB
  - **Test**: `SELECT * FROM audio_segments LIMIT 10`
  - **Expected**: Transcriptions stored

- [ ] **Task 6.4.2**: Test prosody JSON storage
  - **Test**: Verify JSONB field has valid data
  - **Expected**: Prosody features stored correctly

### 6.5 Integration Test
- [ ] **Task 6.5.1**: Run audio processor standalone
  - **Command**: `python services/audio/audio_processor.py test-stream-id`
  - **Test**: Process audio from Redis
  - **Expected**: Transcriptions appear in DB

- [ ] **Task 6.5.2**: Test with real speech
  - **Test**: Process video with clear speech
  - **Expected**: >90% transcription accuracy

---

## PHASE 7: VLM REASONING SERVICE (RunPod API Client)

### 7.1 Service Skeleton
- [ ] **Task 7.1.1**: Install VLM dependencies
  - **Command**: `pip install requests runpod pillow`
  - **Test**: `python -c "import runpod; print('OK')"`
  - **Expected**: Package installed
  - **Note**: No vLLM/transformers locally - inference on RunPod

- [ ] **Task 7.1.2**: Create services/vlm/vlm_reasoner.py
  - **File**: Create service that calls RunPod VLM endpoint
  - **Test**: Script runs
  - **Expected**: No errors

- [ ] **Task 7.1.3**: Initialize RunPod VLM client
  - **Code**: Initialize RunPodClient for VLM endpoint
  - **Test**: Test connection to endpoint
  - **Expected**: Successful connection

### 7.2 Context Management
- [ ] **Task 7.2.1**: Implement context window buffering
  - **Code**: Add frame buffering logic
  - **Test**: Store 100 frames in memory
  - **Expected**: Context window maintains size

- [ ] **Task 7.2.2**: Implement frame sampling
  - **Code**: Sample every Nth frame
  - **Test**: Reduce 30fps to 10fps
  - **Expected**: Correct frame rate reduction

### 7.3 VLM Inference (API Calls)
- [ ] **Task 7.3.1**: Create prompt template
  - **Code**: Design prompt for video understanding
  - **Test**: Generate description for test video
  - **Expected**: Coherent description

- [ ] **Task 7.3.2**: Implement _reason_about_context method (RunPod API)
  - **Code**: Encode frames to base64, send with prompt to RunPod VLM endpoint
  - **Test**: Send frames + prompt to RunPod, parse response
  - **Expected**: Returns description within 3 seconds (including API latency)

- [ ] **Task 7.3.3**: Test emotion detection via VLM on RunPod
  - **Test**: Process video with clear emotions via RunPod
  - **Expected**: VLM identifies emotions correctly through API

### 7.4 Event Storage
- [ ] **Task 7.4.1**: Implement _store_event method
  - **Code**: Insert events into DB
  - **Test**: `SELECT * FROM events ORDER BY start_time DESC LIMIT 10`
  - **Expected**: Events stored with descriptions

- [ ] **Task 7.4.2**: Add event scoring
  - **Code**: Compute clip_worthy_score
  - **Test**: Verify scores are reasonable (0-1)
  - **Expected**: High scores for interesting events

### 7.5 Integration Test
- [ ] **Task 7.5.1**: Run VLM service standalone
  - **Command**: `python services/vlm/vlm_reasoner.py test-stream-id`
  - **Test**: Process video frames from Redis
  - **Expected**: Generates descriptions continuously

- [ ] **Task 7.5.2**: Test memory usage over time
  - **Test**: Run for 10 minutes
  - **Expected**: Memory usage stable (no leaks)

---

## PHASE 8: DOCKER ORCHESTRATION (Lightweight - No GPU Containers)

### 8.1 Dockerfile Creation (CPU-Only)
- [ ] **Task 8.1.1**: Create Dockerfile.base
  - **File**: Lightweight base image with Python only (no CUDA)
  - **Test**: `docker build -f Dockerfile.base -t video-agent-base .`
  - **Expected**: Image builds successfully
  - **Note**: Much smaller image (~2GB) since no GPU libraries

- [ ] **Task 8.1.2**: Create requirements.txt (complete)
  - **File**: Dependencies for API clients, DB, Redis (no PyTorch/CUDA)
  - **Content**: `requests, runpod, opencv-python, pillow, numpy, psycopg2-binary, redis, livekit, supervision, librosa, soundfile, fastapi, uvicorn`
  - **Test**: File is valid
  - **Expected**: All packages listed

- [ ] **Task 8.1.3**: Build base image
  - **Command**: Build and verify size
  - **Test**: `docker images | grep video-agent-base`
  - **Expected**: Image ~2-3GB (much smaller than GPU version)

### 8.2 Service Dockerfiles (CPU-Only)
- [ ] **Task 8.2.1**: Create services/ingestion/Dockerfile
  - **File**: Inherit from base, add service code
  - **Test**: Build ingestion image
  - **Expected**: Build succeeds quickly

- [ ] **Task 8.2.2**: Create services/detection/Dockerfile
  - **File**: Detection service Dockerfile (RunPod API client)
  - **Test**: Build and run test
  - **Expected**: Can call RunPod YOLO endpoint

- [ ] **Task 8.2.3**: Create services/audio/Dockerfile
  - **File**: Audio service Dockerfile (RunPod API client)
  - **Test**: Build and test
  - **Expected**: Can call RunPod Whisper endpoint

- [ ] **Task 8.2.4**: Create services/vlm/Dockerfile
  - **File**: VLM service Dockerfile (RunPod API client)
  - **Test**: Build (fast - no large models)
  - **Expected**: Small image (~2GB), can call RunPod VLM endpoint

### 8.3 Docker Compose (No GPU Resources)
- [ ] **Task 8.3.1**: Create docker-compose.yml
  - **File**: Complete stack without GPU reservations
  - **Content**: postgres, redis, minio, ingestion, detection (API client), audio (API client), vlm (API client)
  - **Test**: `docker-compose config` validates
  - **Expected**: Valid YAML, no GPU device requirements

- [ ] **Task 8.3.2**: Test infrastructure services
  - **Command**: `docker-compose up -d postgres redis minio`
  - **Test**: `docker ps` shows 3 running containers
  - **Expected**: All services healthy

- [ ] **Task 8.3.3**: Test processing services
  - **Command**: `docker-compose up -d detection audio vlm`
  - **Test**: Check logs for RunPod API connection success
  - **Expected**: Services start without crashes, can reach RunPod endpoints

### 8.4 Environment Configuration (RunPod Credentials)
- [ ] **Task 8.4.1**: Create .env template
  - **File**: `.env.example` with all variables
  - **Content**: `RUNPOD_API_KEY`, `RUNPOD_YOLO_ENDPOINT_ID`, `RUNPOD_WHISPER_ENDPOINT_ID`, `RUNPOD_VLM_ENDPOINT_ID`, DB credentials, etc.
  - **Test**: File documents all required env vars
  - **Expected**: Template created

- [ ] **Task 8.4.2**: Create actual .env file
  - **File**: `.env` with real RunPod and DB credentials (not committed)
  - **Test**: Source .env and check all RunPod variables set
  - **Expected**: All vars set, RunPod API keys present

### 8.5 Startup Scripts
- [ ] **Task 8.5.1**: Create start.sh
  - **File**: Main startup script from PRD
  - **Test**: `bash -n start.sh` (syntax check)
  - **Expected**: Script is valid

- [ ] **Task 8.5.2**: Make scripts executable
  - **Command**: `chmod +x start.sh`
  - **Test**: `ls -la start.sh` shows execute permission
  - **Expected**: Executable

- [ ] **Task 8.5.3**: Test startup script
  - **Command**: `./start.sh`
  - **Test**: All services start in order
  - **Expected**: Full stack running

---

## PHASE 9: END-TO-END TESTING

### 9.1 Test Data Preparation
- [ ] **Task 9.1.1**: Download test video
  - **File**: Get sample video with people talking (1-2 min)
  - **Test**: Verify video has audio and clear faces
  - **Expected**: Valid test video

- [ ] **Task 9.1.2**: Create test stream in database
  - **SQL**: Insert into streams table
  - **Test**: `SELECT * FROM streams WHERE stream_name='test'`
  - **Expected**: Stream record exists

### 9.2 Pipeline Test
- [ ] **Task 9.2.1**: Create tests/e2e_test.py
  - **File**: Test script from PRD
  - **Test**: Script is valid Python
  - **Expected**: No syntax errors

- [ ] **Task 9.2.2**: Run ingestion test
  - **Test**: Inject test video frames to Redis
  - **Expected**: Frames appear in Redis streams

- [ ] **Task 9.2.3**: Verify detection service
  - **Test**: Check detections in database
  - **Query**: `SELECT COUNT(*) FROM detections WHERE stream_id='test-id'`
  - **Expected**: Detections > 0

- [ ] **Task 9.2.4**: Verify audio service
  - **Test**: Check transcriptions
  - **Query**: `SELECT text FROM audio_segments WHERE stream_id='test-id' LIMIT 10`
  - **Expected**: Transcribed text visible

- [ ] **Task 9.2.5**: Verify VLM service
  - **Test**: Check event descriptions
  - **Query**: `SELECT description FROM events WHERE stream_id='test-id' LIMIT 10`
  - **Expected**: Events generated

### 9.3 Performance Testing (RunPod Serverless)
- [ ] **Task 9.3.1**: Measure frame processing rate
  - **Test**: Process 1000 frames via RunPod endpoints, measure time
  - **Metric**: Frames per second (target: >15fps with API latency)
  - **Expected**: Near real-time considering network overhead

- [ ] **Task 9.3.2**: Measure API latency
  - **Test**: Time individual RunPod API calls
  - **Metric**: Per-endpoint latency (YOLO: <200ms, Whisper: <500ms, VLM: <2s)
  - **Expected**: Acceptable latency per endpoint

- [ ] **Task 9.3.3**: Measure end-to-end latency
  - **Test**: Time from frame ingestion to event creation
  - **Metric**: End-to-end latency (target: <10s with RunPod)
  - **Expected**: Acceptable latency including all API calls

- [ ] **Task 9.3.4**: Monitor RunPod costs
  - **Test**: Check RunPod dashboard for usage costs
  - **Metric**: Cost per hour of processing
  - **Expected**: Cost-effective compared to dedicated GPU rental

- [ ] **Task 9.3.5**: Monitor local memory usage
  - **Command**: `docker stats`
  - **Test**: Check container memory (should be low without models)
  - **Expected**: <4GB RAM per service, no OOM errors

- [ ] **Task 9.3.6**: Test RunPod auto-scaling
  - **Test**: Send burst of requests, check RunPod scales workers
  - **Expected**: RunPod scales up workers during high load

### 9.4 Data Validation
- [ ] **Task 9.4.1**: Verify data consistency
  - **Test**: Check all tables have data for test stream
  - **Expected**: Data in all tables (frames, detections, audio_segments, events)

- [ ] **Task 9.4.2**: Verify timestamp alignment
  - **Test**: Compare timestamps across tables
  - **Expected**: Timestamps are synchronized

- [ ] **Task 9.4.3**: Verify JSON fields
  - **Test**: Parse JSONB fields (bbox, prosody, etc.)
  - **Expected**: Valid JSON, no null fields

---

## PHASE 10: ADVANCED FEATURES (OPTIONAL)

### 10.1 Emotion Detection Integration
- [ ] **Task 10.1.1**: Research emotion detection models
  - **Action**: Evaluate FG-Net, HTNet, or FER alternatives
  - **Test**: Find pre-trained model
  - **Expected**: Model available

- [ ] **Task 10.1.2**: Create services/emotion/emotion_detector.py
  - **Code**: Service skeleton
  - **Test**: Script runs
  - **Expected**: No errors

- [ ] **Task 10.1.3**: Integrate with detection service
  - **Code**: Add emotion analysis after face detection
  - **Test**: Detect emotions on faces
  - **Expected**: Emotion labels per face

- [ ] **Task 10.1.4**: Store emotions in database
  - **Code**: Insert into emotions table
  - **Test**: `SELECT * FROM emotions LIMIT 10`
  - **Expected**: Emotion data stored

### 10.2 Speaker Diarization
- [ ] **Task 10.2.1**: Load pyannote diarization model
  - **Code**: Initialize pipeline
  - **Test**: Run on test audio
  - **Expected**: Speaker segments identified

- [ ] **Task 10.2.2**: Integrate with audio processor
  - **Code**: Add speaker_id to transcriptions
  - **Test**: Multiple speakers in audio
  - **Expected**: Different speaker IDs assigned

### 10.3 Highlight Generation
- [ ] **Task 10.3.1**: Design highlight scoring algorithm
  - **Logic**: Score based on emotions, speech, events
  - **Test**: Score test events
  - **Expected**: High scores for interesting moments

- [ ] **Task 10.3.2**: Create services/highlights/selector.py
  - **Code**: Query events and score
  - **Test**: Select top 10 highlights
  - **Expected**: Highlights identified

- [ ] **Task 10.3.3**: Implement video clipping
  - **Code**: Use ffmpeg to extract clips
  - **Test**: Export 30-second highlight
  - **Expected**: Video file created

- [ ] **Task 10.3.4**: Upload to S3/MinIO
  - **Code**: Upload clip to storage
  - **Test**: Verify file in MinIO
  - **Expected**: Clip accessible

### 10.4 API Layer
- [ ] **Task 10.4.1**: Create services/api/main.py
  - **File**: FastAPI service from PRD
  - **Test**: `uvicorn services.api.main:app`
  - **Expected**: API server runs

- [ ] **Task 10.4.2**: Test API endpoints
  - **Test**: `curl http://localhost:8000/streams`
  - **Expected**: Returns JSON with streams

- [ ] **Task 10.4.3**: Implement WebSocket events
  - **Code**: Real-time event streaming
  - **Test**: Connect WebSocket client
  - **Expected**: Receives live events

---

## PHASE 11: MONITORING & OBSERVABILITY

### 11.1 Logging
- [ ] **Task 11.1.1**: Configure structured logging
  - **Code**: Use Python logging with JSON formatter
  - **Test**: Check log output format
  - **Expected**: JSON logs

- [ ] **Task 11.1.2**: Centralize logs
  - **Tool**: Set up log aggregation (ELK, Loki, etc.)
  - **Test**: View logs in dashboard
  - **Expected**: All service logs visible

### 11.2 Metrics
- [ ] **Task 11.2.1**: Add Prometheus metrics to services
  - **Code**: Add prometheus_client to each service
  - **Test**: `curl http://localhost:8000/metrics`
  - **Expected**: Metrics endpoint returns data

- [ ] **Task 11.2.2**: Deploy Prometheus
  - **Docker**: Add Prometheus to docker-compose
  - **Test**: Access Prometheus UI
  - **Expected**: Scraping all services

- [ ] **Task 11.2.3**: Create Grafana dashboard
  - **File**: Import dashboard JSON
  - **Test**: View metrics in Grafana
  - **Expected**: Visualizations show data

### 11.3 Alerting
- [ ] **Task 11.3.1**: Configure Prometheus alerts
  - **File**: alerting rules for errors, latency
  - **Test**: Trigger test alert
  - **Expected**: Alert fires

- [ ] **Task 11.3.2**: Set up notification channel
  - **Tool**: Email, Slack, or PagerDuty
  - **Test**: Send test notification
  - **Expected**: Notification received

---

## PHASE 12: PRODUCTION READINESS

### 12.1 Security
- [ ] **Task 12.1.1**: Generate secure passwords
  - **Command**: `openssl rand -base64 32` for each secret
  - **Test**: Update .env with strong passwords
  - **Expected**: No default passwords

- [ ] **Task 12.1.2**: Set up secrets management
  - **Tool**: Use Docker secrets or vault
  - **Test**: Services read secrets correctly
  - **Expected**: Credentials not in code

- [ ] **Task 12.1.3**: Configure firewall
  - **Command**: Restrict ports, allow only necessary access
  - **Test**: `sudo ufw status`
  - **Expected**: Only required ports open

- [ ] **Task 12.1.4**: Enable SSL/TLS
  - **Tool**: Use Let's Encrypt for API
  - **Test**: Access API via HTTPS
  - **Expected**: Valid certificate

### 12.2 Backup & Recovery
- [ ] **Task 12.2.1**: Configure database backups
  - **Script**: pg_dump cron job
  - **Test**: Restore from backup
  - **Expected**: Data restored successfully

- [ ] **Task 12.2.2**: Backup model files
  - **Script**: Sync models to S3
  - **Test**: Verify backup exists
  - **Expected**: Models backed up

- [ ] **Task 12.2.3**: Document recovery procedures
  - **File**: Create RECOVERY.md
  - **Test**: Follow procedures in test environment
  - **Expected**: Can recover from failure

### 12.3 Scaling (RunPod Serverless Auto-Scaling)
- [ ] **Task 12.3.1**: Configure RunPod endpoint scaling settings
  - **Action**: In RunPod dashboard, set min/max workers per endpoint
  - **Config**: Min: 0 (scale to zero), Max: 10 (auto-scale up)
  - **Test**: Observe workers scale based on demand
  - **Expected**: Auto-scaling works, costs minimized during idle

- [ ] **Task 12.3.2**: Test horizontal scaling of local services
  - **Config**: Add replicas in docker-compose for detection/audio/vlm services
  - **Test**: Run 3 detection service replicas making RunPod API calls
  - **Expected**: Load distributed across replicas

- [ ] **Task 12.3.3**: Configure load balancing
  - **Tool**: Add nginx for API load balancing (if using API layer)
  - **Test**: Requests distributed across service replicas
  - **Expected**: Even distribution

- [ ] **Task 12.3.4**: Optimize RunPod endpoint selection
  - **Strategy**: Route to different endpoints based on load or region
  - **Test**: Deploy multiple RunPod endpoints in different regions
  - **Expected**: Lower latency, higher availability

### 12.4 Documentation
- [ ] **Task 12.4.1**: Create deployment guide
  - **File**: DEPLOYMENT.md with step-by-step instructions
  - **Test**: New user can deploy using guide
  - **Expected**: Clear documentation

- [ ] **Task 12.4.2**: Document API
  - **Tool**: OpenAPI/Swagger for API endpoints
  - **Test**: Access API docs
  - **Expected**: All endpoints documented

- [ ] **Task 12.4.3**: Create troubleshooting guide
  - **File**: TROUBLESHOOTING.md with common issues
  - **Test**: Document known issues and solutions
  - **Expected**: Helpful guide

### 12.5 Final Testing
- [ ] **Task 12.5.1**: Full system stress test
  - **Test**: Process 1-hour livestream
  - **Expected**: No crashes, stable performance

- [ ] **Task 12.5.2**: Verify all features working
  - **Checklist**: Test each component end-to-end
  - **Expected**: All features functional

- [ ] **Task 12.5.3**: Production deployment
  - **Action**: Deploy to production environment
  - **Test**: Monitor for 24 hours
  - **Expected**: System stable in production

---

## COMPLETION CHECKLIST

### Core Infrastructure (Must Have)
- [ ] RunPod account created with credits
- [ ] Environment set up (Docker, Python - no GPU needed)
- [ ] Database operational (PostgreSQL + TimescaleDB)
- [ ] Redis and MinIO running
- [ ] All RunPod endpoints deployed and tested

### RunPod Endpoints (Must Have)
- [ ] YOLO detection endpoint deployed and responding
- [ ] Whisper transcription endpoint deployed and responding
- [ ] VLM reasoning endpoint deployed and responding
- [ ] All endpoint credentials stored in .env file
- [ ] Endpoint latency acceptable (<200ms YOLO, <500ms Whisper, <2s VLM)

### Core Services (Must Have)
- [ ] Ingestion service working (can ingest video/audio)
- [ ] Detection service working (calls RunPod YOLO endpoint, tracks objects)
- [ ] Audio service working (calls RunPod Whisper endpoint)
- [ ] VLM service working (calls RunPod VLM endpoint)

### Integration (Must Have)
- [ ] Services communicate via Redis
- [ ] Data stored in PostgreSQL
- [ ] End-to-end test passes
- [ ] Docker Compose orchestrates all services (CPU-only containers)
- [ ] All services can reach RunPod endpoints

### Advanced Features (Nice to Have)
- [ ] Emotion detection endpoint integrated
- [ ] Speaker diarization working
- [ ] Highlight generation functional
- [ ] API layer deployed
- [ ] Multi-region RunPod endpoints for failover

### Production (Must Have for Production)
- [ ] Monitoring and metrics (including RunPod cost tracking)
- [ ] Logging centralized
- [ ] Backups configured
- [ ] Security hardened (RunPod API keys secured)
- [ ] Documentation complete
- [ ] RunPod auto-scaling configured
- [ ] Spending alerts set up in RunPod dashboard

---

## TESTING STRATEGY

**Key Principle**: Test after EVERY task, not just at the end.

### Unit Testing
- Test each function/method in isolation
- Use pytest for Python code
- Mock external dependencies (Redis, DB)

### Integration Testing
- Test service-to-service communication
- Verify data flows through pipeline
- Check Redis streams and DB writes

### End-to-End Testing
- Test complete pipeline with real video
- Measure performance and accuracy
- Validate output quality

### Performance Testing
- Measure throughput (frames/second)
- Check latency (ingestion to output)
- Monitor resource usage (GPU, RAM, CPU)

---

## NOTES

1. **Don't skip tests**: Each verification step is critical. If something doesn't work, fix it before proceeding.

2. **Start small**: Use short test videos (10-30 seconds) initially. Scale up after basics work.

3. **GPU memory**: Watch VRAM usage. You may need to adjust batch sizes or model quantization.

4. **Debugging**: Enable verbose logging during development. Use `docker logs -f <service>` to monitor.

5. **Performance**: Initial focus is correctness, then optimize for speed.

6. **Incremental development**: Build one service at a time, test thoroughly, then move to next.

7. **Document issues**: Keep notes on problems encountered and solutions found.

8. **Version control**: Commit after completing each phase.

---

## ESTIMATED TIME PER PHASE (RunPod Architecture)

- Phase 0 (Pre-flight + RunPod Setup): 1-2 hours
- Phase 1 (Environment Setup - No GPU): 1-2 hours
- Phase 2 (Database): 1-2 hours
- Phase 3 (RunPod Endpoints): 2-3 hours (faster - just deploying endpoints)
- Phase 4 (Ingestion): 4-6 hours
- Phase 5 (Detection API Client): 3-4 hours (simpler than local inference)
- Phase 6 (Audio API Client): 3-4 hours
- Phase 7 (VLM API Client): 4-5 hours
- Phase 8 (Docker - No GPU): 1-2 hours (faster builds)
- Phase 9 (E2E Testing): 2-4 hours
- Phase 10 (Advanced): 6-10 hours
- Phase 11 (Monitoring): 2-4 hours
- Phase 12 (Production): 3-5 hours

**Total estimated time**: 35-55 hours of development work

**Time Savings vs Local GPU**: ~20-30% faster due to:
- No GPU driver/CUDA installation
- No model downloading/storage locally
- Faster Docker builds (smaller images)
- Simpler deployment (no GPU orchestration)

---

---

## RUNPOD SERVERLESS ADVANTAGES

### Cost Benefits
1. **Pay-per-second pricing**: Only pay when models are actively processing
2. **No idle costs**: Auto-scale to zero when not in use
3. **No upfront GPU investment**: No need to buy RTX 4090 or rent dedicated servers
4. **Estimated costs**:
   - YOLO Detection: ~$0.0004 per image (RTX 4090)
   - Whisper ASR: ~$0.001 per minute of audio (A40)
   - VLM Reasoning: ~$0.003 per inference (A100)
   - **Example**: 1 hour of 30fps video processing ≈ $30-50 vs $200-400 for dedicated GPU server

### Operational Benefits
1. **No GPU management**: RunPod handles drivers, CUDA, model loading
2. **Auto-scaling**: Handles traffic spikes automatically
3. **Geographic distribution**: Deploy endpoints in multiple regions for lower latency
4. **Easy upgrades**: Switch to better GPUs without infrastructure changes
5. **Reliability**: RunPod manages uptime, failover, load balancing

### Development Benefits
1. **Faster iteration**: No waiting for model downloads or Docker GPU builds
2. **Laptop-friendly**: Develop on any machine (Mac, Windows, Linux)
3. **Team collaboration**: Everyone uses same endpoints, consistent results
4. **Easy testing**: Spin up/down endpoints as needed

## RUNPOD BEST PRACTICES

### 1. Endpoint Configuration
- **Worker count**: Start with min=0, max=3, increase based on load
- **GPU selection**:
  - YOLO: RTX 4090 or A40 (good price/performance)
  - Whisper: A40 or A100 (faster for audio)
  - VLM: A100 40GB (needs high memory)
- **Timeout**: Set to 60s for VLM, 30s for others
- **Idle timeout**: 5 seconds (balance between cold starts and costs)

### 2. API Call Optimization
- **Batching**: Send multiple frames in one request when possible
- **Image compression**: Use JPEG quality 85-95 (balance size vs quality)
- **Async requests**: Use asyncio to call multiple endpoints in parallel
- **Retry logic**: Implement exponential backoff for failed requests
- **Caching**: Cache repeated requests (e.g., same frame to multiple models)

### 3. Cost Optimization
- **Frame sampling**: Don't send every frame (e.g., 10fps instead of 30fps)
- **Smart triggering**: Only call VLM when interesting events detected
- **Endpoint pooling**: Use multiple cheaper endpoints vs one expensive one
- **Monitor spend**: Set up RunPod spending alerts

### 4. Error Handling
- **Graceful degradation**: If RunPod endpoint fails, log but continue
- **Fallback strategies**: Use simpler models or skip frame if endpoint down
- **Health checks**: Ping endpoints periodically to warm up workers
- **Circuit breaker**: Stop calling failed endpoints temporarily

### 5. Testing Strategy
- **Local mock**: Create mock RunPod responses for local testing
- **Staging endpoints**: Use separate RunPod endpoints for dev/staging/prod
- **Cost tracking**: Monitor RunPod usage during testing to avoid surprises

## RUNPOD ENDPOINT TEMPLATES

You can use these custom handlers when deploying RunPod endpoints:

### YOLO Endpoint Handler Example
```python
# handler.py for RunPod YOLO endpoint
import runpod
from ultralytics import YOLO
import base64
import io
import numpy as np
from PIL import Image

model = YOLO('yolo11m.pt')

def handler(event):
    input_data = event['input']
    image_b64 = input_data['image']

    # Decode image
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))

    # Run inference
    results = model(image)

    # Format output
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'class_name': model.names[int(box.cls[0])]
            })

    return {'detections': detections}

runpod.serverless.start({'handler': handler})
```

### Whisper Endpoint Handler Example
```python
# handler.py for RunPod Whisper endpoint
import runpod
from faster_whisper import WhisperModel
import base64
import io
import numpy as np
import soundfile as sf

model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')

def handler(event):
    input_data = event['input']
    audio_b64 = input_data['audio']

    # Decode audio
    audio_data = base64.b64decode(audio_b64)
    audio, sr = sf.read(io.BytesIO(audio_data))

    # Transcribe
    segments, info = model.transcribe(audio, vad_filter=True)

    # Format output
    transcription = []
    for seg in segments:
        transcription.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text,
            'confidence': seg.avg_logprob
        })

    return {'transcription': transcription, 'language': info.language}

runpod.serverless.start({'handler': handler})
```

---

**Ready to start? Begin with Phase 0: Pre-Flight Checks!**

**Key First Steps**:
1. Create RunPod account
2. Deploy your first endpoint (start with YOLO)
3. Test the endpoint with a sample image
4. Once working, proceed with building the services
