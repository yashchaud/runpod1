# Qwen3-VL-8B Endpoint

For Qwen3-VL-8B, we recommend using RunPod's official vLLM worker template rather than a custom handler.

## Deployment Instructions

### Option 1: Quick Deploy (Recommended)

1. Go to RunPod Console → **Serverless** → **New Endpoint**
2. Under **Quick Deploy**, find **Serverless vLLM**
3. Click **Configure**
4. Set environment variables:

```bash
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
DTYPE=float16
```

5. Select GPU: **A100 40GB** (minimum required)
6. Deploy

### Option 2: Custom Docker Image

Use the official vLLM worker image:

**Docker Image**: `runpod/worker-v1-vllm:stable-cuda12.1.0`

**Environment Variables**:
- `MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct`
- `TOKENIZER_NAME=Qwen/Qwen3-VL-8B-Instruct`
- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.9`
- `DTYPE=float16`

## Usage Example

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

print(result['output'])
```

## Cost Estimate

- **A100 40GB**: ~$0.003-0.005 per inference
- **Processing Time**: 1-3 seconds per frame

## Notes

- Qwen3-VL-8B requires at least 20GB VRAM
- Use A100 40GB or 80GB for best performance
- vLLM provides optimized inference for multimodal models
- Supports batch processing for better efficiency
