# Qwen3-VL Video Processing with vLLM on RunPod

Production-ready serverless deployment of **Qwen3-VL-8B-Instruct** using **vLLM** on RunPod for processing long-form videos with zero-error rate and parallel chunk processing.

## Features

- **Zero-Error Design**: Comprehensive validation, retry logic, and error handling
- **Long Video Support**: Intelligent chunking respecting Qwen3-VL's 768-frame limit
- **Parallel Processing**: Efficient batch processing of video chunks
- **Optimized Performance**: FP8 quantization, vLLM optimizations, adaptive FPS
- **Automated CI/CD**: GitHub Actions for Docker build and deployment
- **Serverless Ready**: RunPod serverless worker with auto-scaling

## Architecture

```
Video Input ’ Validation ’ Chunking ’ Frame Extraction ’ vLLM Inference ’ Aggregation ’ Results
```

### Key Components

1. **Video Processor** ([src/video_processor.py](src/video_processor.py))
   - Intelligent video chunking with configurable duration
   - Automatic chunk overlap for temporal context
   - Adaptive chunk sizing based on video FPS

2. **Frame Extractor** ([src/frame_extractor.py](src/frame_extractor.py))
   - FFmpeg-based frame extraction
   - Automatic FPS calculation to stay within frame budget
   - Support for keyframe-only extraction

3. **vLLM Client** ([src/vllm_client.py](src/vllm_client.py))
   - Qwen3-VL model inference with vLLM
   - Parallel chunk processing with continuous batching
   - Result aggregation and summarization

4. **RunPod Handler** ([src/handler.py](src/handler.py))
   - Serverless function handler
   - Input validation and error handling
   - Temporary file management

## Requirements

### GPU Requirements

| Model Variant | Minimum VRAM | Recommended GPU |
|---------------|--------------|-----------------|
| FP8 (Recommended) | ~16GB | RTX 4090, A5000, L40 |
| BF16 | ~24GB | RTX 6000 Ada, A100 40GB |

### Software Requirements

- CUDA 12.4+
- Python 3.11+
- vLLM >= 0.11.0
- FFmpeg

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/qwen3-vl-runpod.git
cd qwen3-vl-runpod
```

### 2. Set Up GitHub Secrets

Required secrets for GitHub Actions:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `HF_TOKEN`: HuggingFace token (for gated models)

Go to **Settings > Secrets and variables > Actions** and add these secrets.

### 3. Build Docker Image

The GitHub Actions workflow automatically builds and pushes on commits to `main`:

```yaml
# Triggered automatically on push to main
# Or manually via Actions tab > Build and Push Docker Image > Run workflow
```

Manual build:

```bash
docker build -t yourusername/qwen3-vl-runpod:latest .
docker push yourusername/qwen3-vl-runpod:latest
```

### 4. Deploy to RunPod

#### Option A: Using RunPod Console

1. Go to [RunPod Serverless](https://runpod.io/console/serverless)
2. Click **+ New Endpoint**
3. Configure:
   - **Container Image**: `yourusername/qwen3-vl-runpod:latest`
   - **GPU Type**: RTX 4090 or better
   - **Container Disk**: 20 GB
   - **Environment Variables**:
     ```
     MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
     MAX_MODEL_LEN=131072
     GPU_MEMORY_UTILIZATION=0.90
     HF_TOKEN=your_hf_token
     ```
4. Click **Deploy**

#### Option B: Using RunPod API

```python
import runpod

runpod.api_key = "your_runpod_api_key"

endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run({
    "video_url": "https://example.com/video.mp4",
    "prompt": "Describe what happens in this video in detail.",
    "max_tokens": 512,
    "aggregate": True
})

print(result)
```

## Usage

### API Input Format

```json
{
  "video_url": "https://example.com/long_video.mp4",
  "prompt": "Analyze this video and describe all key events.",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "chunk_duration": 60.0,
  "aggregate": true,
  "aggregation_prompt": "Provide a comprehensive summary of the entire video:"
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | **required** | URL to video file |
| `video_path` | string | optional | Local path (alternative to URL) |
| `prompt` | string | **required** | Analysis prompt |
| `max_tokens` | int | 512 | Max tokens per chunk |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.8 | Nucleus sampling |
| `top_k` | int | 20 | Top-k sampling |
| `chunk_duration` | float | 60.0 | Chunk duration in seconds |
| `aggregate` | bool | true | Aggregate chunk results |
| `aggregation_prompt` | string | auto | Custom aggregation prompt |

### Response Format

```json
{
  "success": true,
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration": 180.5,
    "total_frames": 5415
  },
  "num_chunks": 3,
  "chunk_results": [
    {
      "chunk_index": 0,
      "start_time": 0.0,
      "end_time": 60.0,
      "success": true,
      "response": "The video begins with...",
      "processing_time": 5.2
    }
  ],
  "aggregated_response": "This video shows...",
  "total_successful_chunks": 3,
  "total_processing_time": 15.6
}
```

## Configuration

### Environment Variables

Configure via environment variables in RunPod:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-VL-8B-Instruct-FP8` | Model identifier |
| `MAX_MODEL_LEN` | `131072` | Max context length |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory fraction |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs |
| `MAX_FRAMES_PER_CHUNK` | `768` | Max frames per chunk |
| `CHUNK_DURATION` | `60.0` | Default chunk duration |
| `CHUNK_OVERLAP` | `2.0` | Overlap between chunks |
| `HF_TOKEN` | - | HuggingFace token |

### Custom Configuration

Edit [config.yaml](config.yaml) and rebuild Docker image for persistent changes.

## Performance Optimization

### 1. GPU Selection

- **Best Performance**: A100 80GB, H100
- **Cost-Effective**: RTX 4090, L40, A5000
- **Minimum**: RTX 3090, A10G

### 2. vLLM Optimizations

```python
# Enabled by default in Dockerfile
ENV OMP_NUM_THREADS=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 3. Chunk Duration Tuning

For high FPS videos (60fps+):
```json
{
  "chunk_duration": 30.0  # Shorter chunks for more frames
}
```

For low FPS videos (24fps):
```json
{
  "chunk_duration": 90.0  # Longer chunks OK
}
```

## Error Handling

The system implements multiple error handling strategies:

1. **Input Validation**: Video format, duration, codec verification
2. **Chunk Size Calculation**: Automatic adjustment for frame budget
3. **Retry Logic**: Exponential backoff for transient failures
4. **Graceful Degradation**: FPS reduction if needed
5. **Health Checks**: Pre-flight endpoint validation
6. **Timeout Management**: Proper timeouts at each stage

## Development

### Local Testing

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct-FP8"
export HF_TOKEN="your_token"
```

3. **Run locally**:
```bash
cd src
python handler.py
```

### Testing with Sample Video

```python
import requests

response = requests.post(
    "http://localhost:8000/run",
    json={
        "input": {
            "video_url": "https://example.com/test_video.mp4",
            "prompt": "Describe this video."
        }
    }
)

print(response.json())
```

## Monitoring

### Logs

View logs in RunPod console:
- Request/response details
- Processing times per chunk
- Error messages with stack traces

### Metrics

Track in RunPod dashboard:
- Total requests
- Average processing time
- Error rate
- GPU utilization

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `GPU_MEMORY_UTILIZATION` or `MAX_MODEL_LEN`

```
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=65536
```

### Issue: Slow Processing

**Solution**:
- Use FP8 quantized model
- Reduce chunk duration
- Enable tensor parallelism for multi-GPU

### Issue: Frame Budget Exceeded

**Solution**: System automatically adjusts FPS, but you can manually configure:

```json
{
  "chunk_duration": 30.0,  // Shorter chunks
  "max_frames_per_chunk": 600  // Reduce frame limit
}
```

### Issue: Video Download Timeout

**Solution**: Use `video_path` with pre-uploaded video or increase timeout in handler.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{qwen3vl_runpod,
  title = {Qwen3-VL Video Processing on RunPod},
  year = {2025},
  url = {https://github.com/yourusername/qwen3-vl-runpod}
}
```

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen3-VL) for the amazing Qwen3-VL model
- [vLLM Project](https://github.com/vllm-project/vllm) for the inference engine
- [RunPod](https://runpod.io) for serverless GPU infrastructure

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/qwen3-vl-runpod/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/qwen3-vl-runpod/discussions)
- RunPod Support: [RunPod Discord](https://discord.gg/runpod)

---

**Built with d for the AI community**
