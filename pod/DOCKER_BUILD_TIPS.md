# Docker Build Tips - Space Issues

## Problem: "No space left on device"

Building the pod Docker image locally requires **significant disk space** due to PyTorch and transformers:
- PyTorch: ~2GB
- Transformers: ~1GB
- Build cache: ~5-10GB
- Temporary files: ~5GB

**Total space needed**: ~15-20GB free

## Solution 1: Use GitHub Actions (Recommended) ⭐

Let GitHub build the image for you - it's **free, faster, and has unlimited space**:

### Step 1: Push to GitHub
```bash
git add pod/
git commit -m "Add pod"
git push origin main
```

### Step 2: Wait for Build
- Go to your repo → Actions tab
- Wait 5-10 minutes for build to complete
- Image automatically pushed to Docker Hub

### Step 3: Deploy
Use the built image: `your-dockerhub-username/video-processor:latest`

**Benefits**:
- ✅ No local disk space needed
- ✅ 3× faster builds (GitHub has better hardware)
- ✅ Automatic caching
- ✅ Free for public repos

## Solution 2: Clean Up Docker (Local Build)

If you must build locally, free up space first:

### Check Space
```bash
docker system df
```

### Clean Up
```bash
# Remove unused containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove build cache
docker buildx prune -a -f

# Nuclear option: Remove everything
docker system prune -a --volumes -f
```

This can free **10-50GB** of space!

### Then Build
```bash
cd pod
docker build -t your-username/video-processor:latest .
docker push your-username/video-processor:latest
```

## Solution 3: Build on RunPod

Build directly on RunPod (uses their disk space):

### Step 1: Create Pod
- GPU: Any (even CPU pod works for building)
- Disk: 50GB
- Docker Image: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`

### Step 2: SSH and Build
```bash
# SSH into pod
ssh root@your-pod-id.runpod.net

# Clone repo
git clone https://github.com/your-username/runpod.git
cd runpod/pod

# Build
docker build -t your-username/video-processor:latest .
docker login
docker push your-username/video-processor:latest
```

### Step 3: Stop Pod
No need to keep it running after build.

## Why Image is Large

### Original Dockerfile (Removed)
```dockerfile
# This was downloading 16GB model during build!
RUN python3 -c "from transformers import Qwen2VLForConditionalGeneration..."
```

### Optimized Dockerfile (Current)
```dockerfile
# Model downloads on first startup instead
# Reduces image from ~30GB to ~10GB
```

**Model download**: Happens automatically when pod starts (1-2 minutes)

## Build Space Requirements

| Method | Disk Space Needed | Build Time | Cost |
|--------|------------------|------------|------|
| **GitHub Actions** | **0 (remote)** | **5-10 min** | **Free** ⭐ |
| Local (clean) | 15-20 GB | 10-15 min | Free |
| Local (with junk) | 30-40 GB | 10-15 min | Free |
| RunPod Build Pod | 0 (remote) | 10-15 min | $0.10 |

## Troubleshooting

### "No space left on device" During Build

**Check available space**:
```bash
# Windows
wmic logicaldisk get size,freespace,caption

# Linux/Mac
df -h
```

**Need**: At least 20GB free on the drive where Docker stores images.

**Fix**: Clean up Docker (see Solution 2) or use GitHub Actions (see Solution 1)

### "Docker daemon not running"

**Windows**:
1. Start Docker Desktop
2. Wait for it to fully start (green icon)
3. Try again

**Linux**:
```bash
sudo systemctl start docker
```

### Build is Very Slow

**Use GitHub Actions** - their servers are much faster:
- Your machine: 10-15 minutes
- GitHub Actions: 5-7 minutes
- Plus it runs in the background while you work!

## Recommended Workflow

### For Development
Use GitHub Actions for builds:

```bash
# Make changes
nano pod/adaptive_processor.py

# Commit and push (triggers automatic build)
git add pod/
git commit -m "Update processor"
git push

# GitHub builds in background
# Check Actions tab for progress

# Once built, deploy to RunPod
# Image: your-username/video-processor:latest
```

### For Testing
Use pre-built image from Docker Hub:

```bash
# Don't build locally - just use existing image
docker pull your-username/video-processor:latest

# Or deploy directly to RunPod with that image
```

## Space Savings

| Version | Image Size | Build Space | Model Location |
|---------|-----------|-------------|----------------|
| Old (pre-download) | ~30 GB | ~40 GB | In image |
| **New (on-startup)** | **~10 GB** | **~15 GB** | Downloaded on pod start |

**Savings**: 66% smaller image, 62% less build space!

**Trade-off**: 1-2 minute model download on first pod startup (one-time)

## Summary

**Best approach**: Use GitHub Actions (free, fast, no local space needed)

1. Push code to GitHub
2. GitHub builds automatically
3. Deploy from Docker Hub
4. Model downloads on first startup

**Alternative**: Clean Docker and build locally if needed

**Avoid**: Building with full Docker cache and old images

See [pod/README.md](README.md) for deployment instructions.
