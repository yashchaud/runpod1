# Troubleshooting Guide

Common issues and their solutions when deploying Qwen3-VL on RunPod.

## Docker Build Issues

### Issue: "No space left on device" during Docker build

**Symptoms**:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**Solutions**:

#### Solution 1: Use Dockerfile.simple (Recommended)
```bash
docker build -f Dockerfile.simple -t yourusername/qwen3-vl-runpod:latest .
```

This uses a pre-built base image and requires less disk space.

#### Solution 2: Free up Docker disk space
```bash
# Remove unused images
docker system prune -a --volumes

# Remove build cache
docker builder prune -a

# Check available space
df -h
```

#### Solution 3: Increase Docker disk size
- **Docker Desktop (Mac/Windows)**: Settings > Resources > Disk image size > Increase to 64GB
- **Linux**: Increase disk space on host or mount larger volume

#### Solution 4: Use GitHub Actions
GitHub Actions has been configured to automatically free up disk space before building. Just push to main:
```bash
git add .
git commit -m "Trigger build"
git push origin main
```

The workflow uses `Dockerfile.simple` by default which requires less disk space.

### Issue: GitHub Actions build fails

**Symptoms**:
```
ERROR: failed to solve: process did not complete successfully
```

**Solutions**:

1. **Check GitHub Secrets are set**:
   - Go to Settings > Secrets and variables > Actions
   - Ensure these are set:
     - `DOCKERHUB_USERNAME`
     - `DOCKERHUB_TOKEN`
     - `HF_TOKEN` (optional but recommended)

2. **Manually trigger with Dockerfile.simple**:
   - Go to Actions tab
   - Click "Build and Push Docker Image"
   - Click "Run workflow"
   - Select "Dockerfile.simple" from dropdown
   - Click "Run workflow"

3. **Build locally and push manually**:
   ```bash
   # Build
   docker build -f Dockerfile.simple -t yourusername/qwen3-vl-runpod:latest .

   # Login
   docker login

   # Push
   docker push yourusername/qwen3-vl-runpod:latest
   ```

### Issue: Undefined variable warning

**Symptoms**:
```
UndefinedVar: Usage of undefined variable '$PYTHONPATH'
```

**Solution**: This is just a warning and can be ignored. The build should still succeed. If it bothers you, the Dockerfile has been updated to use `${PYTHONPATH}` instead.

## RunPod Deployment Issues

### Issue: Container fails to start

**Symptoms**: Worker shows "Failed" status immediately after starting.

**Solutions**:

1. **Check container image exists**:
   - Verify image is public on Docker Hub
   - Or provide Docker Hub credentials in RunPod

2. **Check environment variables**:
   ```
   MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
   HF_TOKEN=your_token_here
   ```

3. **Check container disk size**:
   - Set to at least 20GB
   - Model weights are ~9GB + dependencies

4. **Check logs**:
   - Go to RunPod endpoint > Logs
   - Look for error messages

### Issue: Model fails to download

**Symptoms**:
```
Error: Failed to download model from HuggingFace
```

**Solutions**:

1. **Set HF_TOKEN**:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
   Get token from: https://huggingface.co/settings/tokens

2. **Check model name**:
   ```
   MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
   ```

3. **Verify model exists**:
   - Visit: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
   - Ensure you have access

### Issue: GPU Out of Memory (OOM)

**Symptoms**:
```
CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions**:

1. **Reduce GPU memory utilization**:
   ```
   GPU_MEMORY_UTILIZATION=0.80
   ```

2. **Reduce context length**:
   ```
   MAX_MODEL_LEN=65536
   ```

3. **Reduce frames per chunk**:
   ```
   MAX_FRAMES_PER_CHUNK=512
   ```

4. **Use larger GPU**:
   - RTX 4090 (24GB) - Minimum for FP8
   - A100 40GB - Better performance
   - A100 80GB - Best performance

5. **Ensure using FP8 model**:
   ```
   MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
   ```
   Not the BF16 version which requires more VRAM.

### Issue: Request timeout

**Symptoms**:
```
Error: Request timed out after 600s
```

**Solutions**:

1. **Increase execution timeout**:
   - RunPod endpoint settings > Execution Timeout > 900s (15 min)

2. **Reduce chunk duration** (process smaller segments):
   ```json
   {
     "chunk_duration": 30.0
   }
   ```

3. **Split very long videos**:
   - For 2+ hour videos, split into smaller files
   - Process separately

### Issue: Video download fails

**Symptoms**:
```
Error: Failed to download video from URL
```

**Solutions**:

1. **Check URL is accessible**:
   ```bash
   curl -I https://your-video-url.mp4
   ```

2. **Use direct file URL** (not webpage):
   - ✅ Good: `https://example.com/video.mp4`
   - ❌ Bad: `https://youtube.com/watch?v=xxx`

3. **Upload to cloud storage**:
   - AWS S3 (generate presigned URL)
   - Google Cloud Storage
   - Direct download link

4. **Use local path** (if video on RunPod network storage):
   ```json
   {
     "video_path": "/runpod-volume/videos/video.mp4"
   }
   ```

## Processing Issues

### Issue: Video validation fails

**Symptoms**:
```
Error: Video validation failed: Invalid or zero duration
```

**Solutions**:

1. **Check video file is valid**:
   ```bash
   ffmpeg -i video.mp4
   ```

2. **Supported formats**:
   - MP4 (recommended)
   - AVI
   - MOV
   - MKV

3. **Re-encode video**:
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
   ```

### Issue: Poor quality results

**Symptoms**: Model generates irrelevant or low-quality descriptions.

**Solutions**:

1. **Improve prompt**:
   ```json
   {
     "prompt": "Describe this video in detail, focusing on: 1) Main actions and events, 2) Objects and people visible, 3) Text or signage, 4) Setting and atmosphere"
   }
   ```

2. **Increase max_tokens**:
   ```json
   {
     "max_tokens": 1024
   }
   ```

3. **Adjust temperature**:
   ```json
   {
     "temperature": 0.7,
     "top_p": 0.8,
     "top_k": 20
   }
   ```

4. **Use shorter chunks** for better temporal resolution:
   ```json
   {
     "chunk_duration": 30.0
   }
   ```

### Issue: Slow processing

**Symptoms**: Processing takes much longer than expected.

**Solutions**:

1. **Use FP8 model** (if not already):
   ```
   MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
   ```

2. **Optimize chunk duration**:
   ```json
   {
     "chunk_duration": 60.0
   }
   ```

3. **Use faster GPU**:
   - H100 > A100 > RTX 4090 > L40

4. **Reduce frame extraction**:
   ```
   MAX_FRAMES_PER_CHUNK=512
   ```

## Local Testing Issues

### Issue: Import errors when testing locally

**Symptoms**:
```python
ModuleNotFoundError: No module named 'vllm'
```

**Solutions**:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Python path**:
   ```bash
   export PYTHONPATH=./src:$PYTHONPATH
   ```

3. **Use virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

### Issue: FFmpeg not found

**Symptoms**:
```
RuntimeError: FFmpeg not found or not working
```

**Solutions**:

1. **Install FFmpeg**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Mac
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Verify installation**:
   ```bash
   ffmpeg -version
   ```

## Performance Optimization

### Slow chunk processing

**Expected Performance**:
- RTX 4090: ~5-15s per 60s chunk
- A100 40GB: ~3-10s per 60s chunk

**If slower**:

1. Check GPU utilization:
   - Should be 85-95% during processing
   - Lower = potential bottleneck

2. Verify FP8 model is being used

3. Check for CPU bottlenecks (shouldn't be significant)

4. Ensure SSD storage (not HDD) for frame extraction

### High costs

**Cost Reduction Strategies**:

1. **Use scale-to-zero**:
   - Set min workers = 0
   - Set idle timeout = 5-10s

2. **Use cheaper GPU**:
   - RTX 4090: ~$0.40/hr
   - L40: ~$0.60/hr
   - A5000: ~$0.50/hr

3. **Optimize chunk duration**:
   - Longer chunks = fewer model calls
   - But less granular results

4. **Batch processing**:
   - Process multiple videos in one session
   - Amortize cold start time

## Getting Help

If your issue isn't listed here:

1. **Check logs**: Most issues show clear error messages in logs

2. **GitHub Issues**: https://github.com/yourusername/qwen3-vl-runpod/issues

3. **RunPod Discord**: https://discord.gg/runpod

4. **vLLM Discussions**: https://github.com/vllm-project/vllm/discussions

## Quick Reference

### Minimum Requirements
- GPU: RTX 4090 or better (24GB+ VRAM)
- Container Disk: 20GB+
- Model: Qwen3-VL-8B-Instruct-FP8

### Recommended Settings
```env
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.90
MAX_FRAMES_PER_CHUNK=768
CHUNK_DURATION=60.0
```

### API Request Example
```json
{
  "video_url": "https://example.com/video.mp4",
  "prompt": "Describe this video in detail.",
  "max_tokens": 512,
  "temperature": 0.7,
  "chunk_duration": 60.0,
  "aggregate": true
}
```
