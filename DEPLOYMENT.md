# Deployment Guide: Qwen3-VL on RunPod

Complete step-by-step guide to deploy Qwen3-VL-8B with vLLM on RunPod serverless infrastructure.

## Prerequisites

1. **GitHub Account** with repository access
2. **Docker Hub Account** (or alternative registry)
3. **RunPod Account** with credits
4. **HuggingFace Account** with token

## Step 1: Prepare GitHub Repository

### 1.1 Create Repository

```bash
git init
git add .
git commit -m "Initial commit: Qwen3-VL RunPod deployment"
git branch -M main
git remote add origin https://github.com/yourusername/qwen3-vl-runpod.git
git push -u origin main
```

### 1.2 Configure GitHub Secrets

Navigate to **Settings > Secrets and variables > Actions > New repository secret**

Add the following secrets:

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `DOCKERHUB_USERNAME` | your_username | Docker Hub login |
| `DOCKERHUB_TOKEN` | your_token | Docker Hub access token |
| `HF_TOKEN` | your_hf_token | HuggingFace API token |

**Get Docker Hub Token:**
1. Go to [Docker Hub](https://hub.docker.com/)
2. Account Settings > Security > New Access Token
3. Copy the token

**Get HuggingFace Token:**
1. Go to [HuggingFace](https://huggingface.co/settings/tokens)
2. Create new token with read access
3. Copy the token

## Step 2: Build and Push Docker Image

### Option A: Automatic Build (Recommended)

GitHub Actions will automatically build on push to `main`:

```bash
git add .
git commit -m "Trigger Docker build"
git push origin main
```

Monitor build at: `https://github.com/yourusername/qwen3-vl-runpod/actions`

### Option B: Manual Build

```bash
# Build
docker build -t yourusername/qwen3-vl-runpod:latest .

# Login to Docker Hub
docker login

# Push
docker push yourusername/qwen3-vl-runpod:latest
```

### Verify Image

Check Docker Hub: `https://hub.docker.com/r/yourusername/qwen3-vl-runpod`

## Step 3: Deploy to RunPod

### 3.1 Create Serverless Endpoint

1. Go to [RunPod Serverless](https://runpod.io/console/serverless)
2. Click **+ New Endpoint**

### 3.2 Configure Endpoint

**Basic Settings:**
- **Endpoint Name**: `qwen3-vl-video-processor`
- **Select GPU**: RTX 4090 / A5000 / L40 (minimum 24GB VRAM recommended)
- **Container Image**: `yourusername/qwen3-vl-runpod:latest`
- **Container Disk**: `20 GB`

**Advanced Settings:**

**Environment Variables:**
```
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.90
TENSOR_PARALLEL_SIZE=1
HF_TOKEN=your_hf_token_here
MAX_FRAMES_PER_CHUNK=768
CHUNK_DURATION=60.0
CHUNK_OVERLAP=2.0
```

**Scaling Configuration:**
- **Idle Timeout**: 5 seconds
- **Min Workers**: 0 (scale to zero)
- **Max Workers**: 3 (adjust based on budget)
- **GPUs Per Worker**: 1

**Execution Timeout**: 600 seconds (10 minutes)

### 3.3 Deploy

Click **Deploy** and wait for:
- ✓ Container pulled
- ✓ Worker initialized
- ✓ Endpoint ready

**Expected initialization time**: 3-5 minutes (model download + loading)

## Step 4: Test Deployment

### 4.1 Get Endpoint Details

From RunPod console, note:
- **Endpoint ID**: `xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- **API Key**: Found in Settings > API Keys

### 4.2 Test with RunPod API

```python
import runpod

runpod.api_key = "your_api_key"

endpoint = runpod.Endpoint("your_endpoint_id")

# Test with sample video
result = endpoint.run_sync({
    "video_url": "https://download.samplelib.com/mp4/sample-5s.mp4",
    "prompt": "Describe this video."
})

print(result)
```

### 4.3 Test with REST API

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://download.samplelib.com/mp4/sample-5s.mp4",
      "prompt": "Describe what you see."
    }
  }'
```

### 4.4 Expected Response

```json
{
  "id": "sync-xxxxx",
  "status": "COMPLETED",
  "output": {
    "success": true,
    "video_info": {
      "duration": 5.0,
      "fps": 30.0,
      "width": 1280,
      "height": 720
    },
    "num_chunks": 1,
    "aggregated_response": "The video shows..."
  }
}
```

## Step 5: Monitor and Optimize

### 5.1 Monitor Metrics

RunPod Dashboard shows:
- **Request Rate**: Requests per minute
- **Execution Time**: Average processing time
- **Error Rate**: Failed requests percentage
- **GPU Utilization**: GPU usage metrics
- **Costs**: Real-time cost tracking

### 5.2 Performance Optimization

**If experiencing OOM (Out of Memory):**

```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTILIZATION=0.85

# Reduce context length
MAX_MODEL_LEN=65536

# Reduce chunk size
MAX_FRAMES_PER_CHUNK=512
```

**If processing too slow:**

```bash
# Use shorter chunks for parallel processing
CHUNK_DURATION=30.0

# For multi-GPU setups
TENSOR_PARALLEL_SIZE=2
```

**For better quality:**

```bash
# Use BF16 model (requires more VRAM)
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct

# Increase max tokens
# (in API request, not env var)
"max_tokens": 1024
```

### 5.3 Cost Optimization

**Strategies:**
1. **Scale to Zero**: Set min workers to 0
2. **Idle Timeout**: Keep at 5-10 seconds
3. **GPU Selection**: RTX 4090 is most cost-effective for this model
4. **Batch Processing**: Use async endpoint for multiple videos

**Estimated Costs (as of 2025):**
- RTX 4090: ~$0.40/hour
- A5000: ~$0.50/hour
- A100 40GB: ~$1.50/hour

For 60-second video:
- Processing time: ~15-30 seconds
- Cost per video: ~$0.002-$0.005

## Step 6: Production Considerations

### 6.1 Error Handling

The system includes built-in error handling, but monitor:
- Video validation failures
- Download timeouts
- Model inference errors

**Set up alerts** in RunPod for:
- Error rate > 5%
- Average execution time > 60s
- GPU errors

### 6.2 Rate Limiting

Configure in RunPod:
- **Max Concurrency**: 30 (default)
- Adjust based on GPU capacity

### 6.3 Logging

Access logs via RunPod console:
- **Worker Logs**: Real-time execution logs
- **Request Logs**: All API requests
- **Error Logs**: Failed requests with traces

### 6.4 Security

- **Never commit** API keys or tokens
- Use **environment variables** for all secrets
- Enable **IP whitelisting** if needed (RunPod settings)
- Implement **request signing** for production

## Step 7: Continuous Deployment

### 7.1 Update Deployment

To deploy new version:

```bash
# Make changes
git add .
git commit -m "Update: improved video chunking"
git push origin main

# GitHub Actions builds new image automatically
# Tag: yourusername/qwen3-vl-runpod:main-<git-sha>
```

Update RunPod endpoint:
1. Go to endpoint settings
2. Update container image to new tag
3. Click **Save & Deploy**

### 7.2 Rollback

If issues arise:

1. Go to Docker Hub
2. Find previous working tag
3. Update RunPod endpoint to previous image
4. Redeploy

### 7.3 A/B Testing

Deploy two endpoints:
- **Production**: Stable version
- **Canary**: New version

Route 10% traffic to canary, monitor metrics, then switch.

## Troubleshooting

### Issue: Container Fails to Start

**Check:**
- Docker image exists and is public (or credentials provided)
- Container disk size is sufficient (20GB minimum)
- Environment variables are correct

**Solution:**
```bash
# Verify image locally
docker pull yourusername/qwen3-vl-runpod:latest
docker run --rm -it --gpus all yourusername/qwen3-vl-runpod:latest python -c "import vllm; print(vllm.__version__)"
```

### Issue: Model Download Fails

**Check:**
- HF_TOKEN is valid and has read access
- Model name is correct: `Qwen/Qwen3-VL-8B-Instruct-FP8`
- Network connectivity in RunPod

**Solution:**
- Verify token at https://huggingface.co/settings/tokens
- Check model exists: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8

### Issue: Request Timeout

**Adjust timeout:**
- RunPod endpoint settings > Execution Timeout > 900s (15 min)
- For very long videos, consider splitting into smaller segments

### Issue: GPU Out of Memory

**Reduce memory usage:**
```bash
GPU_MEMORY_UTILIZATION=0.80
MAX_MODEL_LEN=65536
MAX_FRAMES_PER_CHUNK=512
```

Or upgrade GPU:
- RTX 4090 (24GB) → A100 40GB
- A100 40GB → A100 80GB

## Next Steps

1. **Integrate** into your application
2. **Monitor** performance metrics
3. **Optimize** costs and latency
4. **Scale** based on demand

## Support

- **GitHub Issues**: https://github.com/yourusername/qwen3-vl-runpod/issues
- **RunPod Discord**: https://discord.gg/runpod
- **vLLM Docs**: https://docs.vllm.ai

---

**Deployment Complete! 🎉**

Your Qwen3-VL video processing endpoint is now live and ready to handle production traffic.
