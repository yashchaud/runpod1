# Video Processing Pod - Adaptive Qwen3-VL-8B

Intelligent video processing pod that automatically scales based on available VRAM. Perfect for processing long Google Meet recordings, screen shares, and app demonstrations.

## Features

- **Adaptive Scaling**: Automatically adjusts batch size, concurrency, and frame sampling based on available VRAM (24GB, 48GB, 80GB+)
- **Intelligent Frame Sampling**: Scene change detection ensures key moments are captured
- **Latest Model**: Qwen3-VL-8B-Instruct (October 15, 2025) with 32-language OCR and 256K context
- **Efficient Processing**: Processes 1-hour videos in ~10-20 minutes (depending on VRAM)
- **FastAPI Server**: RESTful API for job submission and tracking
- **Multiple Analysis Modes**:
  - `screen_share`: Google Meet/Zoom recordings with app demos
  - `ui_detection`: UI element extraction with positions
  - `meeting_analysis`: Video call participant and activity tracking
  - `app_demo`: Application demonstration analysis

## VRAM Scaling Examples

| VRAM | Batch Size | Concurrent | Frame Rate | 1hr Video Time |
|------|-----------|-----------|-----------|----------------|
| 24GB | 8         | 4         | 1/30fps   | ~15-20 min     |
| 48GB | 16        | 6         | 1/15fps   | ~8-12 min      |
| 80GB | 24        | 8         | 1/10fps   | ~5-8 min       |

## Deployment on RunPod

### 1. Build Docker Image

```bash
# Navigate to pod directory
cd pod

# Build image
docker build -t video-processor:latest .

# Push to Docker Hub (or RunPod Container Registry)
docker tag video-processor:latest your-username/video-processor:latest
docker push your-username/video-processor:latest
```

### 2. Create RunPod Pod

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Click "Deploy"
3. Select GPU:
   - **RTX 4090 (24GB)**: Good for moderate workloads
   - **RTX A6000 (48GB)**: Better for faster processing
   - **A100 (40/80GB)**: Best for high-volume processing
4. Container Settings:
   - **Docker Image**: `your-username/video-processor:latest`
   - **Expose HTTP Ports**: `8000`
   - **Container Disk**: 50GB (for model storage)
5. Click "Deploy On-Demand" or "Deploy Spot"

### 3. Get Pod URL

After deployment, you'll get a URL like:
```
https://your-pod-id-8000.proxy.runpod.net
```

Save this URL - you'll use it to submit videos.

## Usage

### Option 1: Using Client Script (Recommended)

```bash
# Install dependencies
pip install requests

# Process video
python scripts/process_video_pod.py \
    "your-video.mp4" \
    --pod-url "https://your-pod-id-8000.proxy.runpod.net" \
    --mode screen_share \
    --output results.json
```

### Option 2: Using cURL

```bash
# Submit video
curl -X POST \
    -F "video=@your-video.mp4" \
    -F "mode=screen_share" \
    https://your-pod-id-8000.proxy.runpod.net/process

# Returns: {"job_id": "abc-123", "status": "pending"}

# Check status
curl https://your-pod-id-8000.proxy.runpod.net/jobs/abc-123

# Get result
curl https://your-pod-id-8000.proxy.runpod.net/results/abc-123 > result.json
```

### Option 3: Python Requests

```python
import requests

pod_url = "https://your-pod-id-8000.proxy.runpod.net"

# Submit video
with open("video.mp4", "rb") as f:
    response = requests.post(
        f"{pod_url}/process",
        files={"video": f},
        data={"mode": "screen_share"}
    )

job_id = response.json()["job_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"{pod_url}/jobs/{job_id}").json()
    if status["status"] in ["completed", "failed"]:
        break
    print(f"Status: {status['status']}")
    time.sleep(10)

# Get result
if status["status"] == "completed":
    result = requests.get(f"{pod_url}/results/{job_id}").json()
    print(result)
```

## Analysis Modes

### screen_share (Default)
Best for Google Meet recordings, Zoom screen shares, presentation recordings.

**Output includes:**
- Scene description (what's being shared)
- Screen layout and UI positions
- All visible text (UI labels, chat, code)
- Interactive elements with positions
- User activity and context

### ui_detection
Focused on extracting UI elements with precise positions.

**Output includes:**
- UI element types (button, input, menu)
- Text labels
- Approximate coordinates
- Element states

### meeting_analysis
Analyzes video call dynamics.

**Output includes:**
- Participant count
- Screen sharing detection
- Chat messages
- Current discussion topic

### app_demo
Specialized for application demonstrations.

**Output includes:**
- Application identification
- Screen/page names
- Feature descriptions
- User action tracking

## API Endpoints

### GET /health
Health check and GPU status
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_info": {
    "device_count": 1,
    "devices": [{"name": "NVIDIA RTX 4090", "memory_allocated_gb": 15.2}]
  }
}
```

### GET /config
Get current processor configuration
```json
{
  "config": {
    "batch_size": 8,
    "max_concurrent_batches": 4,
    "total_vram_gb": 24.0,
    "precision": "bfloat16",
    "frame_sample_rate": 30
  }
}
```

### POST /process
Submit video for processing

**Form Data:**
- `video`: Video file (multipart/form-data)
- `mode`: Analysis mode (screen_share, ui_detection, meeting_analysis, app_demo)

**Response:**
```json
{
  "job_id": "abc-123",
  "status": "pending",
  "message": "Job queued for processing"
}
```

### GET /jobs/{job_id}
Get job status
```json
{
  "job_id": "abc-123",
  "status": "processing",
  "video_filename": "video.mp4",
  "mode": "screen_share",
  "created_at": "2025-01-22T10:00:00",
  "started_at": "2025-01-22T10:01:00",
  "progress": 45.0
}
```

### GET /results/{job_id}
Download processing result (JSON file)

### GET /jobs
List all jobs

### DELETE /jobs/{job_id}
Delete job and associated files

## Output Format

```json
{
  "video_info": {
    "fps": 30.0,
    "total_frames": 108000,
    "duration_seconds": 3600,
    "width": 1920,
    "height": 1080
  },
  "processing_config": {
    "batch_size": 8,
    "concurrent_batches": 4,
    "frame_sample_rate": 30,
    "precision": "bfloat16"
  },
  "processing_stats": {
    "total_frames_analyzed": 3600,
    "processing_time_minutes": 15.2,
    "frames_per_second": 3.95,
    "efficiency_ratio": 0.25
  },
  "results": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "is_keyframe": false,
      "analysis": {
        "scene_description": "Google Meet screen share showing a React application demo",
        "layout": "Split screen: left shows app interface, right shows participant grid",
        "text_extraction": ["Settings", "User Profile", "Dashboard", "Sign Out"],
        "ui_elements": [
          {
            "type": "button",
            "text": "Sign Out",
            "position": "top-right",
            "state": "enabled"
          }
        ],
        "activity": "Presenter demonstrating user authentication flow",
        "context": "Explaining how login and session management works"
      }
    }
  ]
}
```

## Cost Optimization

### Spot Instances
Use RunPod Spot instances for 70% cost savings. Jobs continue even if pod restarts (state saved to disk).

### Auto-Scaling
Pod automatically adjusts to available VRAM:
- If VRAM increases → larger batches + more concurrency
- If VRAM decreases → smaller batches + less concurrency

### Frame Sampling Intelligence
- Short videos (<10 min): Sample every 10 frames (3fps @ 30fps)
- Medium videos (10-60 min): Sample every 30 frames (1fps)
- Long videos (>1 hour): Sample every 60 frames (0.5fps)
- Scene changes: Always captured regardless of sampling

## Performance Tips

1. **Use appropriate GPU**: RTX 4090 (24GB) is sweet spot for cost/performance
2. **Batch multiple videos**: Submit several videos at once for better GPU utilization
3. **Choose right mode**: `screen_share` is comprehensive but slower; `ui_detection` is faster if you only need UI elements
4. **Adjust frame sampling**: Longer videos use lower frame rates automatically
5. **Use Spot instances**: 70% cheaper, jobs resume if interrupted

## Monitoring

### Check Pod Health
```bash
curl https://your-pod-id-8000.proxy.runpod.net/health
```

### View Logs
In RunPod console → Select pod → Logs tab

### GPU Utilization
```bash
# SSH into pod
nvidia-smi -l 1
```

## Troubleshooting

### "Processor not initialized"
Wait 1-2 minutes after pod starts - model is loading (16GB download + initialization)

### Out of Memory
Pod will automatically reduce batch size on next video. For immediate fix:
- Use lower resolution video
- Or upgrade to larger GPU

### Slow Processing
- Check GPU utilization with `nvidia-smi`
- Verify concurrent batches aren't hitting memory limits
- Consider upgrading to faster GPU (A100)

### Job Failed
Check job details:
```bash
curl https://your-pod-id-8000.proxy.runpod.net/jobs/{job_id}
```

Error message will indicate issue.

## Local Testing

Test locally before deploying:

```bash
# Build image
docker build -t video-processor:latest .

# Run container
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/test_videos:/workspace/uploads \
    -v $(pwd)/results:/workspace/results \
    video-processor:latest

# Submit test video
python scripts/process_video_pod.py \
    test_video.mp4 \
    --pod-url http://localhost:8000
```

## Architecture

```
Client (your machine)
    ↓ (upload video)
FastAPI Server (port 8000)
    ↓ (queue job)
Background Task Manager
    ↓ (parallel processing)
Adaptive Processor
    ├─ VRAM Detector (auto-configure)
    ├─ Frame Sampler (intelligent sampling)
    └─ Qwen3-VL-8B (scene understanding)
    ↓ (save result)
Results Storage
    ↓ (download)
Client (receives JSON)
```

## Why This Approach?

### vs YOLO
YOLO detects 80 generic objects (person, laptop, tv). For Google Meet recordings showing app demos, it's completely useless - it can't read text, understand UI, or recognize applications.

Qwen3-VL-8B:
- Reads all visible text (OCR)
- Understands screen content
- Detects UI elements with positions
- Recognizes applications and features
- Provides context and semantic understanding

### vs Serverless Functions
Serverless endpoints (like RunPod serverless) have:
- Cold start delays (5-30s)
- Fixed 1.2s per request overhead
- No batch optimization
- Higher cost per minute

Persistent Pod:
- No cold starts
- Parallel batch processing (4-8 batches simultaneously)
- Auto-scales to available VRAM
- Lower cost for long-running tasks

**Example**: 1-hour video
- Serverless: ~30-40 minutes processing
- Pod (24GB): ~15-20 minutes processing
- Pod (80GB): ~5-8 minutes processing

## License

MIT

## Support

Issues? Contact [your-email] or open GitHub issue.
