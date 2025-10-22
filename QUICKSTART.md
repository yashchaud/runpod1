# Quick Start - Video Processing with RunPod

## Prerequisites
1. RunPod account with API key
2. YOLO and Whisper endpoints deployed on RunPod
3. Python 3.8+ installed
4. FFmpeg installed

## Setup (One-time)

### 1. Install Python dependencies
```bash
pip install requests opencv-python pillow numpy python-dotenv
```

### 2. Configure your `.env` file
Make sure your `.env` file has these values set:
```env
RUNPOD_API_KEY=your-api-key-here
RUNPOD_YOLO_ENDPOINT_ID=your-yolo-endpoint-id
RUNPOD_WHISPER_ENDPOINT_ID=your-whisper-endpoint-id
```

## Process a Video (One Command)

### Windows:
```bash
# Process with default settings (1 frame per second)
process.bat "video.mp4"

# Process with custom frame rate (2 fps)
process.bat "video.mp4" 2

# Process with custom output directory
process.bat "video.mp4" 1 results
```

### Linux/Mac:
```bash
# Process with default settings (use -u for unbuffered output)
python -u scripts/process_video.py "video.mp4"

# Process with custom frame rate (2 fps)
python -u scripts/process_video.py "video.mp4" 2

# Process with custom output directory
python -u scripts/process_video.py "video.mp4" 1 results
```

**Note:** The `-u` flag ensures you see logs in real-time (unbuffered output)

## What It Does

1. **Extracts frames** from your video (at specified fps rate)
2. **Detects objects** in each frame using YOLO endpoint on RunPod
3. **Extracts audio** from the video
4. **Transcribes audio** using Whisper endpoint on RunPod
5. **Saves results** to JSON file in output directory

## Output

Results are saved to: `output/<video_name>_results.json`

The JSON file contains:
- **detections**: Frame-by-frame object detection results
  - timestamp, frame number, bounding boxes, confidence scores
- **transcription**: Audio transcription with timestamps
  - start/end times, text segments, language detected

## Example

```bash
# Process the video in your root directory
process.bat "2025-07-20 15-11-29.mp4"
```

This will:
- Extract ~574 frames (for a 9.5 minute video at 1 fps)
- Detect objects in each frame via RunPod YOLO endpoint (~25-30ms per frame)
- Transcribe all audio via RunPod Whisper endpoint
- Save results to: `output/2025-07-20 15-11-29_results.json`

## Performance

- **YOLO detection**: ~25-30ms per frame (after warmup)
- **Whisper transcription**: ~5-10 seconds for the full audio
- **Total time**: Depends on video length and fps setting

For a 10-minute video at 1 fps:
- ~600 frames × 30ms = ~18 seconds for detection
- ~10 seconds for transcription
- **Total: ~30-40 seconds**

## Troubleshooting

### "RUNPOD_API_KEY not set"
Make sure your `.env` file exists and has the correct API key.

### "ffmpeg not found"
Install ffmpeg from: https://ffmpeg.org/download.html

### Endpoint timeout
If processing very long videos, increase the timeout in the script or process in smaller chunks.

## Cost Estimation

Based on RunPod serverless pricing:
- YOLO (RTX 4090): ~$0.0004 per image
- Whisper (A40): ~$0.001 per minute of audio

For a 10-minute video at 1 fps:
- 600 frames × $0.0004 = $0.24
- 10 minutes × $0.001 = $0.01
- **Total: ~$0.25**
